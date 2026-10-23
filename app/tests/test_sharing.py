"""Security and provenance checks for portable experiment summaries."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import file_io
from app.sharing import ReportStore
from app.model_config import make_model_config
from app.search_summary import validate_search_summary


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


def search_summary(**changes):
    value = {
        "status": "completed", "planned": 4, "completed": 2,
        "best_candidate": "extra", "validation_reason": "Одинаковые проверочные строки",
        "candidates": [
            {"id": "baseline", "label": "Стартовая модель", "model": "HistGradientBoostingRegressor",
             "status": "completed", "metric": 50.0, "selected": False, "elapsed_seconds": 1.0, "error": None},
            {"id": "extra", "label": "ExtraTrees", "model": "ExtraTreesRegressor",
             "status": "completed", "metric": 43.25, "selected": True, "elapsed_seconds": 2.0, "error": None},
            {"id": "failed", "label": "Изменённая модель", "model": "ExtraTreesRegressor",
             "status": "failed", "metric": None, "selected": False, "elapsed_seconds": 0.2,
             "error": "private-key from C:\\Users\\Private\\train.csv"},
            {"id": "skipped", "label": "Последняя модель", "model": "ExtraTreesRegressor",
             "status": "skipped", "metric": None, "selected": False, "elapsed_seconds": 0.0, "error": None},
        ],
    }
    value.update(changes)
    return value


def classification_job(**changes):
    from agent.diagnostics import classification_diagnostics

    diagnostics = classification_diagnostics(["NA", "NA", "Positive", "Positive"],
                                           ["NA", "Positive", "Positive", "Positive"])
    value = job(action="baseline", model="RandomForestClassifier",
                metric={"name": "Macro F1", "value": diagnostics["macro_f1"], "higher_is_better": True},
                split={"seed": 42, "validation_fraction": 0.2, "stratified": True,
                       "validation_rows": 4, "validation_hash": "c" * 64},
                classification_diagnostics=diagnostics)
    value.update(changes)
    return value


class SharingTests(unittest.TestCase):
    def test_fixed_recipe_keeps_feature_processing_but_no_executable_model_artifact(self):
        from agent.features import normalize_feature_config
        features = normalize_feature_config({"signed_log": True, "missing_indicators": True})
        original = job(feature_config=features, training_device="cpu",
                       model_artifact={"available": True, "filename": "untrusted.joblib"})
        report = self.envelope(original)["reports"][0]
        self.assertEqual(report["feature_config"], features)
        self.assertEqual(report["training_device"], "cpu")
        self.assertNotIn("model_artifact", report)
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, self.envelope(original)),
                         {"imported": 1, "skipped": 0})

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

    def test_v3_research_exchange_keeps_provenance_without_machine_reasons_or_local_verification(self):
        from app.tests.test_research_summary import v3_report
        from app.research_summary import research_summary

        summary = research_summary(v3_report())
        summary["stages"][0]["reason"] = "/private/secret-key"
        original = job(action="agent", model="Qwen", metric=None, split=None,
                       comparison_key=None, research=summary,
                       agent_status="fallback", agent_reason="context_budget")
        envelope = self.envelope(original)
        self.assertNotIn("/private", json.dumps(envelope))
        self.store.import_reports("regression", FINGERPRINT, envelope)
        imported, = self.store.list("regression")
        self.assertEqual(imported["research"]["protocol_version"], 3)
        self.assertEqual(imported["agent_reason"], "context_budget")
        self.assertFalse(imported["research"]["evaluations"]["final_check"]["unbiased_estimate"])
        self.assertFalse(imported["verified"])
        self.assertEqual(self.envelope(imported)["reports"], envelope["reports"])

    def test_class_diagnostics_survive_exchange_without_becoming_verified(self):
        original = classification_job()
        envelope = self.envelope(original)
        self.assertEqual(self.store.import_reports("classification", FINGERPRINT, envelope),
                         {"imported": 1, "skipped": 0})
        imported, = self.store.list("classification")
        self.assertEqual(imported["classification_diagnostics"], original["classification_diagnostics"])
        self.assertIs(imported["verified"], False)
        self.assertIs(imported["published"], False)
        self.assertEqual(imported["source"], "imported")
        reexported = self.store.export("classification", FINGERPRINT, [imported])
        self.assertEqual(reexported["reports"], envelope["reports"])
        self.assertEqual(self.store.import_reports("classification", FINGERPRINT, reexported),
                         {"imported": 0, "skipped": 1})

    def test_diagnostic_matrix_changes_report_identity_at_the_same_macro_f1(self):
        from agent.diagnostics import classification_diagnostics

        first = classification_job()
        second = classification_job(classification_diagnostics=classification_diagnostics(
            ["NA", "NA", "Positive", "Positive"], ["NA", "NA", "NA", "Positive"]))
        self.assertAlmostEqual(first["metric"]["value"], second["classification_diagnostics"]["macro_f1"])
        for record in (first, second):
            envelope = self.envelope(record)
            self.assertEqual(self.store.import_reports("classification", FINGERPRINT, envelope),
                             {"imported": 1, "skipped": 0})
        self.assertEqual(len({row["id"] for row in self.store.list("classification")}), 2)

    def test_invalid_class_diagnostics_and_metric_mismatches_never_write_imports(self):
        valid = self.envelope(classification_job())
        mutations = [
            lambda r: r["classification_diagnostics"]["confusion_matrix"][0].__setitem__(0, 100),
            lambda r: r["classification_diagnostics"]["confusion_matrix"][0].__setitem__(0, True),
            lambda r: r["classification_diagnostics"].update(accuracy=float("nan")),
            lambda r: r["classification_diagnostics"].update(rows=99),
            lambda r: r["classification_diagnostics"].update(path="/private/.env"),
            lambda r: r["classification_diagnostics"]["per_class"][0].update(f1=1.0),
            lambda r: r["metric"].update(value=0.123),
            lambda r: r["metric"].update(name="RMSE", higher_is_better=False),
            lambda r: r["split"].update(validation_rows=5),
            lambda r: r.update(action="agent"),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                envelope = deepcopy(valid)
                mutation(envelope["reports"][0])
                with self.assertRaises(ValueError):
                    self.store.import_reports("classification", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_bounded_large_class_report_fits_without_removing_envelope_size_limit(self):
        from agent.diagnostics import classification_diagnostics
        from app import sharing

        labels = ["К" * 190 + str(index) for index in range(50)]
        diagnostics = classification_diagnostics(labels, labels)
        original = classification_job(classification_diagnostics=diagnostics,
                                      metric={"name": "Macro F1", "value": 1.0, "higher_is_better": True},
                                      split={"validation_rows": 50})
        envelope = self.envelope(original)
        self.assertGreater(len(json.dumps(envelope, ensure_ascii=False).encode("utf-8")), 16 * 1024)
        self.assertEqual(self.store.import_reports("classification", FINGERPRINT, envelope),
                         {"imported": 1, "skipped": 0})
        imported, = self.store.list("classification")
        self.assertEqual(imported["classification_diagnostics"], diagnostics)
        with patch.object(sharing, "MAX_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "МБ"):
                self.envelope(original)
        with patch.object(sharing, "MAX_RECORD_BYTES", 1024):
            with self.assertRaisesRegex(ValueError, "КБ"):
                self.envelope(original)

    @patch("app.sharing.MAX_BYTES", 1024 * 1024)
    def test_many_full_class_matrices_roundtrip_near_the_envelope_byte_limit(self):
        from agent.diagnostics import classification_diagnostics
        from app import sharing

        labels = [str(index) for index in range(50)]
        diagnostics = classification_diagnostics(labels, labels)
        original = classification_job(classification_diagnostics=diagnostics,
                                      metric={"name": "Macro F1", "value": 1.0, "higher_is_better": True},
                                      split={"validation_rows": 50})
        single = self.envelope(original)
        record_bytes = len(sharing._encoded(single["reports"][0]))
        overhead = len(sharing._encoded(single)) - record_bytes
        count = min(sharing.MAX_REPORTS, (sharing.MAX_BYTES - overhead + 1) // (record_bytes + 1))
        jobs = [{**original, "id": f"{index:032x}"} for index in range(count)]
        envelope = self.store.export("classification", FINGERPRINT, jobs)
        size = len(sharing._encoded(envelope))
        self.assertGreater(size, sharing.MAX_BYTES * .98)
        self.assertLessEqual(size, sharing.MAX_BYTES)
        self.assertEqual(self.store.import_reports("classification", FINGERPRINT, envelope),
                         {"imported": count, "skipped": 0})
        imported = self.store.list("classification")
        self.assertEqual(len(imported), count)
        reexported = self.store.export("classification", FINGERPRINT, imported)
        self.assertEqual(sorted(reexported["reports"], key=lambda row: row["id"]),
                         sorted(envelope["reports"], key=lambda row: row["id"]))
        self.assertEqual(self.store.import_reports("classification", FINGERPRINT, reexported),
                         {"imported": 0, "skipped": count})
        with self.assertRaisesRegex(ValueError, "МБ"):
            self.store.export("classification", FINGERPRINT, jobs + [{**original, "id": f"{count:032x}"}])

    def test_automl_candidates_and_winner_survive_exchange_without_machine_diagnostics(self):
        config = make_model_config("extra_trees", "regression")
        original = job(action="automl", model_config=config, search=search_summary())
        envelope = self.envelope(original)
        serialized = json.dumps(envelope)
        self.assertNotIn("private-key", serialized)
        self.assertNotIn("Users", serialized)
        self.assertEqual(original["search"]["candidates"][2]["error"],
                         "private-key from C:\\Users\\Private\\train.csv")
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                         {"imported": 1, "skipped": 0})
        imported, = self.store.list("regression")
        self.assertEqual(imported["search"], envelope["reports"][0]["search"])
        self.assertEqual(imported["search"]["best_candidate"], "extra")
        self.assertEqual(imported["model_config"], config)
        self.assertFalse(imported["verified"])
        self.assertFalse(imported["published"])
        reexported = self.envelope(imported)
        self.assertEqual(reexported["reports"], envelope["reports"])
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, reexported),
                         {"imported": 0, "skipped": 1})

    def test_automl_attempt_evidence_changes_identity(self):
        for score in (50.0, 51.0):
            search = search_summary()
            search["candidates"][0]["metric"] = score
            envelope = self.envelope(job(action="automl", search=search,
                                         model_config=make_model_config("extra_trees", "regression")))
            self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                             {"imported": 1, "skipped": 0})
        self.assertEqual(len({row["id"] for row in self.store.list("regression")}), 2)

    def test_invalid_automl_winner_and_evidence_rejected_before_import_writes(self):
        valid = self.envelope(job(action="automl", search=search_summary(),
                                  model_config=make_model_config("extra_trees", "regression")))
        mutations = [
            lambda row: row["search"].update(best_candidate="absent"),
            lambda row: row["search"].update(completed=1),
            lambda row: row["search"].update(planned=99),
            lambda row: row["search"].update(path="/private/.env"),
            lambda row: row["search"]["candidates"][0].update(selected=True),
            lambda row: row["search"]["candidates"][0].update(metric=42),
            lambda row: row["search"]["candidates"][0].update(metric=float("nan")),
            lambda row: row["search"]["candidates"][0].update(id="extra"),
            lambda row: row["search"]["candidates"][0].update(error="x" * 2001),
            lambda row: row["search"]["candidates"][1].update(model="RandomForestClassifier"),
            lambda row: row["search"]["candidates"][1].update(status="failed"),
            lambda row: row["metric"].update(value=44),
            lambda row: row.pop("model_config"),
            lambda row: row.pop("search"),
            lambda row: row.update(action="extra_trees"),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                envelope = deepcopy(valid)
                mutation(envelope["reports"][0])
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_expanded_search_preserves_inner_scores_negative_r2_and_features(self):
        from agent.features import normalize_feature_config
        config = make_model_config("linear", "regression")
        features = normalize_feature_config({"missing_indicators": True})
        winner = {"id": "linear", "label": "Ridge", "model": "Ridge", "status": "completed",
                  "metric": -0.2, "selected": True, "elapsed_seconds": 2.0, "error": None,
                  "selection_metric": -0.1, "verification_metric": -0.2, "phase": "refit",
                  "model_config": config, "feature_config": features,
                  "rejection_reason": None, "resources": {"available": False, "unavailable_reason": "private path"},
                  "durations": {"fit_seconds": 0.2}}
        inner_only = {**winner, "id": "inner", "selected": False, "metric": None,
                      "verification_metric": None, "phase": "selection", "selection_metric": 0.8,
                      "rejection_reason": "not_selected"}
        search = {"status": "completed", "phase": "finished", "planned": 16, "completed": 2,
                  "selection_completed": 2, "verified": 1, "best_candidate": "linear",
                  "validation_reason": "Внешняя проверка отделена от подбора", "candidates": [winner, inner_only]}
        original = job(action="automl", model="Ridge", model_config=config, feature_config=features,
                       metric={"name": "R2", "value": -0.2, "higher_is_better": True}, search=search)
        envelope = self.envelope(original)
        self.store.import_reports("regression", FINGERPRINT, envelope)
        saved, = self.store.list("regression")
        self.assertEqual(saved["feature_config"], features)
        self.assertEqual(saved["search"]["candidates"][1]["selection_metric"], 0.8)
        self.assertIsNone(saved["search"]["candidates"][1]["metric"])
        self.assertNotIn("private path", json.dumps(saved))
        for mutate in (
            lambda row: row["search"]["candidates"][0]["resources"].update(command="execute"),
            lambda row: row["search"]["candidates"][0]["feature_config"].update(path="secret"),
            lambda row: row["search"].update(verified=2),
            lambda row: row["feature_config"].update(signed_log=True),
            lambda row: row["search"]["candidates"][1].update(selected=True),
        ):
            changed = deepcopy(envelope)
            mutate(changed["reports"][0])
            with self.assertRaises(ValueError):
                self.store.import_reports("regression", FINGERPRINT, changed)

    def test_fixed_blend_has_bounded_components_and_retains_features(self):
        from agent.features import normalize_feature_config
        component = make_model_config("linear", "regression")
        config = {"model": "FixedBlend", "parameters": {"weight": 0.5, "components": [component, component]}}
        features = normalize_feature_config()
        row = {"id": "blend", "label": "Среднее двух моделей", "model": "FixedBlend", "status": "completed",
               "metric": 43.25, "selected": True, "elapsed_seconds": 1, "error": None,
               "model_config": config, "feature_config": features, "component_feature_configs": [features, features]}
        search = {"status": "completed", "planned": 1, "completed": 1, "best_candidate": "blend",
                  "validation_reason": "Проверка", "candidates": [row]}
        original = job(action="automl", model="FixedBlend", model_config=config, search=search,
                       feature_config=features, component_feature_configs=[features, features])
        envelope = self.envelope(original)
        self.store.import_reports("regression", FINGERPRINT, envelope)
        saved, = self.store.list("regression")
        self.assertEqual(saved["model_config"], config)
        self.assertEqual(saved["component_feature_configs"], [features, features])
        invalid = deepcopy(envelope)
        invalid["reports"][0]["model_config"]["parameters"]["weight"] = 0.7
        with self.assertRaises(ValueError):
            self.store.import_reports("regression", FINGERPRINT, invalid)

    def test_rejected_blend_does_not_become_eligible_after_component_refit_fails(self):
        ridge = make_model_config("linear", "regression")
        trees = make_model_config("extra_trees", "regression")
        blend = {"model": "FixedBlend", "parameters": {"weight": 0.5, "components": [trees, ridge]}}
        base = {"label": "Проверенная модель", "elapsed_seconds": 1, "error": None,
                "status": "completed", "selected": False}
        summary = {"status": "completed", "planned": 3, "completed": 2,
                   "best_candidate": "fallback", "validation_reason": "Отложенная проверка",
                   "candidates": [
                       {**base, "id": "strongest", "model": "ExtraTreesRegressor", "model_config": trees,
                        "metric": 1.0, "status": "failed", "rejection_reason": "refit_failed"},
                       {**base, "id": "fallback", "model": "Ridge", "model_config": ridge,
                        "metric": 2.0, "selected": True},
                       {**base, "id": "blend", "model": "FixedBlend", "model_config": blend,
                        "metric": 1.5, "rejection_reason": "not_improved"},
                   ]}
        metric = {"name": "RMSE", "value": 2.0, "higher_is_better": False}
        self.assertEqual(validate_search_summary(summary, metric=metric, model="Ridge"), summary)
        # The exemption is specific to a rejected blend; it must not hide a
        # better eligible individual model or a blend that passed selection.
        for model, config, reason in (("Ridge", ridge, "not_improved"),
                                      ("FixedBlend", blend, None)):
            forged = deepcopy(summary)
            forged["candidates"][2].update(model=model, model_config=config, rejection_reason=reason)
            with self.assertRaisesRegex(ValueError, "лучшую проверочную метрику"):
                validate_search_summary(forged, metric=metric, model="Ridge")

    def test_search_progress_can_be_valid_before_first_winner_but_not_final(self):
        summary = search_summary(status="running", completed=0, best_candidate=None)
        for row in summary["candidates"]:
            row.update(status="pending", selected=False, metric=None, error=None)
        summary["candidates"][0]["status"] = "running"
        validated = validate_search_summary(summary, allow_running=True)
        self.assertEqual(validated, summary)
        self.assertIsNot(validated["candidates"], summary["candidates"])
        with self.assertRaises(ValueError):
            validate_search_summary(summary)
        summary["status"] = "completed"
        with self.assertRaises(ValueError):
            validate_search_summary(summary, allow_running=True)

    def test_failed_search_progress_preserves_failure_with_or_without_a_winner(self):
        summary = search_summary(status="failed")
        self.assertEqual(validate_search_summary(summary, allow_running=True), summary)
        with self.assertRaises(ValueError):
            validate_search_summary(summary)
        summary.update(completed=0, best_candidate=None)
        for row in summary["candidates"]:
            row.update(status="failed", selected=False, metric=None, error="Training failed")
        self.assertEqual(validate_search_summary(summary, allow_running=True), summary)
        with self.assertRaises(ValueError):
            validate_search_summary(summary)

    def test_selected_model_parameters_survive_import_reexport_and_deduplication(self):
        config = make_model_config("extra_trees", "regression", {"n_estimators": 200, "max_depth": 12})
        envelope = self.envelope(job(model_config=config))
        self.assertEqual(envelope["reports"][0]["model_config"], config)
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, envelope),
                         {"imported": 1, "skipped": 0})
        imported, = self.store.list("regression")
        self.assertEqual(imported["model_config"], config)
        self.assertEqual(imported["comparison_key"], job()["comparison_key"])
        self.assertIs(imported["verified"], False)
        self.assertIs(imported["published"], False)
        reexported = self.store.export("regression", FINGERPRINT, [imported])
        self.assertEqual(reexported["reports"], envelope["reports"])
        self.assertEqual(self.store.import_reports("regression", FINGERPRINT, reexported),
                         {"imported": 0, "skipped": 1})

    def test_changed_model_parameters_change_report_identity(self):
        for trees in (200, 201):
            config = make_model_config("extra_trees", "regression", {"n_estimators": trees})
            self.assertEqual(self.store.import_reports("regression", FINGERPRINT,
                                                      self.envelope(job(model_config=config))),
                             {"imported": 1, "skipped": 0})
        reports = self.store.list("regression")
        self.assertEqual(len({report["id"] for report in reports}), 2)
        self.assertEqual({report["model_config"]["parameters"]["n_estimators"] for report in reports},
                         {200, 201})

    def test_legacy_and_agent_reports_do_not_gain_default_model_parameters(self):
        for original in (job(), job(action="agent", model=None, metric=None)):
            with self.subTest(action=original["action"]):
                envelope = self.envelope(original)
                self.assertNotIn("model_config", envelope["reports"][0])
                summary = envelope["reports"][0]
                digest = hashlib.sha256(json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                self.store.import_reports("regression", FINGERPRINT, envelope)
                imported = next(report for report in self.store.list("regression")
                                if report["id"] == "import-" + digest)
                self.assertNotIn("model_config", imported)
                reexported = self.store.export("regression", FINGERPRINT, [imported])
                self.assertEqual(reexported["reports"], envelope["reports"])

    def test_malformed_model_parameters_are_rejected_on_export_and_import(self):
        valid = make_model_config("extra_trees", "regression")
        invalid = [None, True, [], {}, {**valid, "path": "/private/.env"},
                   {**valid, "model": "<script>unknown</script>"},
                   {**valid, "parameters": None}, {**valid, "parameters": []},
                   {**valid, "parameters": {"n_estimators": 200}},
                   {**valid, "parameters": {**valid["parameters"], "code": "do not execute"}}]
        for field, values in (("n_estimators", (True, "200", 200.5, float("nan"), 1, 601)),
                              ("max_depth", (False, "12", 100)),
                              ("max_features", (True, float("nan"), "<img src=x>"))):
            invalid.extend({**valid, "parameters": {**valid["parameters"], field: value}} for value in values)
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.envelope(job(model_config=config))
                envelope = self.envelope()
                envelope["reports"][0]["model_config"] = deepcopy(config)
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)
        self.assertFalse((self.root / ".workspace").exists())

    def test_model_metadata_must_match_the_recipe_and_declared_model(self):
        config = make_model_config("extra_trees", "regression")
        for change in ({"action": "agent"}, {"action": "baseline"},
                       {"model": "RandomForestClassifier"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.envelope(job(model_config=config, **change))
                envelope = self.envelope(job(model_config=config))
                envelope["reports"][0].update(change)
                with self.assertRaises(ValueError):
                    self.store.import_reports("regression", FINGERPRINT, envelope)
        # A failed run can retain chosen settings before a fitted model exists.
        pending_model = self.envelope(job(status="failed", model=None, model_config=config))
        self.assertEqual(pending_model["reports"][0]["model_config"], config)
        self.assertIsNone(pending_model["reports"][0]["model"])
        self.assertFalse((self.root / ".workspace").exists())

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

    def test_unknown_metric_fallback_preserves_quality_warning_in_exchange(self):
        envelope = self.envelope(job(action="agent", metric=None, agent_status="fallback",
                                     agent_reason="host_metric_unsupported"))
        self.store.import_reports("regression", FINGERPRINT, envelope)
        imported, = self.store.list("regression")
        self.assertEqual(imported["agent_status"], "fallback")
        self.assertEqual(imported["agent_reason"], "host_metric_unsupported")
        self.assertIsNone(imported["metric"])
        self.assertEqual(self.store.export("regression", FINGERPRINT, [imported])["reports"],
                         envelope["reports"])

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
            {"reports": [job()] * 201}, {"unused": "x" * (8 * 1024 * 1024)},
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

    def test_failed_atomic_import_cleans_temp_and_preserves_existing_reports(self):
        self.store.import_reports("regression", FINGERPRINT, self.envelope())
        existing = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        error = PermissionError(13, "Target is locked")
        error.winerror = 5
        with patch.object(file_io, "WINDOWS", True), patch.object(file_io, "REPLACE_ATTEMPTS", 2), \
                patch.object(file_io.os, "replace", side_effect=error) as replace, \
                patch.object(file_io.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "сохранить") as caught:
                self.store.import_reports("regression", FINGERPRINT,
                                          self.envelope(job(hypothesis="A different experiment")))
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(replace.call_count, 2)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*.json")}, existing)
        self.assertEqual(len(self.store.list("regression")), 1)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_import_rechecks_paths_after_a_windows_lock(self):
        original_safe = self.store._safe
        changed = {"path": None}
        error = PermissionError(13, "Target is locked")
        error.winerror = 32

        def check_path(path):
            if path == changed["path"]:
                raise ValueError("Destination became a link")
            return original_safe(path)

        def locked_replace(source, destination):
            changed["path"] = destination
            raise error

        with patch.object(self.store, "_safe", side_effect=check_path), \
                patch.object(file_io, "WINDOWS", True), \
                patch.object(file_io.os, "replace", side_effect=locked_replace) as replace, \
                patch.object(file_io.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "became a link"):
                self.store.import_reports("regression", FINGERPRINT, self.envelope())
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(self.store.list("regression"), [])
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

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
