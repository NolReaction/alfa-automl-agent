"""XGBoost candidate contracts; native checks run only where xgboost is installed."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import xgboost_candidate as xgb_candidate
from agent.features import FeatureTransformer
from agent.metrics import get_metric, score_predictions
from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
from agent.splits import split_rows
from agent.task_spec import TaskSpec


CLASSES = ["NA", "class c", "01"]
FEATURES = ["x", "period", "kind"]
NATIVE = importlib.util.find_spec("xgboost") is not None
TIME = {"strategy": "time", "time_column": "period"}


def data(*, classification=False, classes=CLASSES, grouped=False, rows=240):
    random = np.random.default_rng(11)
    base = random.normal(size=rows // 2 if grouped else rows)
    if grouped:
        base = np.repeat(base, 2)
    x = ["" if index % 23 == 0 else f"{value:.4f}" for index, value in enumerate(base)]
    periods = 1 + np.arange(len(base)) * 4 // len(base)
    kinds = np.asarray(["NA", "0007", "7", ""])[np.arange(len(base)) // (2 if grouped else 1) % 4]
    frame = pd.DataFrame({"x": x, "period": [str(value) for value in periods], "kind": kinds},
                         index=[f"fit_{index}" for index in range(len(base))])
    if not grouped:
        # Seen only in the latest period: the time-policy held rows.
        frame.loc[frame["period"].eq("4") & frame["kind"].eq("0007"), "kind"] = "held_only"
    if classification:
        cuts = np.quantile(base, np.linspace(0, 1, len(classes) + 1)[1:-1])
        target = pd.Series(np.asarray(classes)[np.searchsorted(cuts, base)], index=frame.index)
    else:
        target = pd.Series(3 * base + np.where(kinds == "7", 2.0, 0.0)
                           + random.normal(scale=.1, size=len(base)), index=frame.index)
    prediction = pd.DataFrame({"x": ["-2.5", "", "2.5"], "period": ["5"] * 3,
                               "kind": ["predict_only", "NA", "7"]},
                              index=["predict_0", "predict_1", "predict_2"])
    return frame, target, prediction


def context(recipe, *, classes=(), phase="inner_selection", deadline=None, positive_class=None):
    metric = get_metric(recipe["metric"])
    return {"version": 3, "protocol_version": 3, "phase": phase, "nonce": "xgboost-unit",
            "recipe": recipe, "code_hash": "a" * 64, "recipe_hash": "b" * 64,
            "features": list(FEATURES), "numeric": ["x", "period"], "categorical": ["kind"],
            "classes": list(classes), "positive_class": positive_class,
            "metric": metric.to_dict(), "task_type": metric.task_type,
            "prediction_kind": metric.prediction_kind,
            "deadline_monotonic": time.monotonic() + 120 if deadline is None else deadline}


def recipe_for(metric="RMSE", validation=None, **kwargs):
    task_type = get_metric(metric).task_type
    validation = validation or (TIME if task_type == "regression" else {"strategy": "random"})
    kwargs.setdefault("max_iterations", 200)
    kwargs.setdefault("stopping_rounds", 10)
    return xgb_candidate.make_xgboost_recipe(task_type, metric, validation=validation, **kwargs)


@contextmanager
def recording():
    calls = {"feature_fit": [], "train": []}

    class RecordingFeatures(FeatureTransformer):
        def fit(self, frame, target=None):
            calls["feature_fit"].append(frame.copy(deep=True))
            return super().fit(frame, target)

    real_train = xgb_candidate._train

    def train(*args, **kwargs):
        calls["train"].append({"rounds": args[3], "early": kwargs.get("held") is not None})
        return real_train(*args, **kwargs)

    with patch.object(xgb_candidate, "FeatureTransformer", RecordingFeatures), \
            patch.object(xgb_candidate, "_train", side_effect=train):
        yield calls


class XGBoostRecipeTests(unittest.TestCase):
    def test_factory_returns_detached_normalized_recipe(self):
        parameters = dict(xgb_candidate.FIXED_CONFIGURATIONS["depth4_regularized_v1"])
        validation, features = dict(TIME), {"missing_indicators": True}
        before = deepcopy((parameters, validation, features))
        recipe = xgb_candidate.make_xgboost_recipe(
            "regression", "rmse", validation=validation, parameters=parameters,
            feature_config=features, max_iterations=300, stopping_rounds=20)
        self.assertEqual((parameters, validation, features), before)
        self.assertEqual((recipe["version"], recipe["family"], recipe["metric"]),
                         (1, "xgboost_native_v1", "RMSE"))
        self.assertEqual(recipe["runtime"], {"package": "xgboost", "package_version": "2.1.4",
                                             "tree_method": "hist", "device": "cpu",
                                             "max_threads": 8, "seed": 42})
        recipe["parameters"]["max_depth"] = 9
        self.assertEqual(parameters, before[0])
        self.assertEqual(json.loads(json.dumps(recipe)), xgb_candidate._checked_recipe(
            json.loads(json.dumps(recipe))))

    def test_both_fixed_configurations_and_partial_whitelist_normalize(self):
        for name, values in xgb_candidate.FIXED_CONFIGURATIONS.items():
            with self.subTest(name=name):
                self.assertEqual(recipe_for(parameters=values)["parameters"], values)
        recipe = recipe_for(parameters={"max_depth": 3, "learning_rate": 0.1})
        self.assertEqual(recipe["parameters"]["max_depth"], 3)
        self.assertIsInstance(recipe["parameters"]["min_child_weight"], float)

    def test_losses_follow_metric_without_renaming_proxies(self):
        expected = {"RMSE": "squared_error", "MSE": "squared_error", "R2": "squared_error",
                    "MAE": "absolute_error", "Macro F1": "log_loss", "ROC AUC": "log_loss",
                    "LogLoss": "log_loss", "Accuracy": "log_loss", "Weighted F1": "log_loss"}
        for metric, loss in expected.items():
            with self.subTest(metric=metric):
                self.assertEqual(recipe_for(metric)["loss"], loss)
        self.assertEqual(xgb_candidate._relation(get_metric("Macro F1")), "proxy_not_host_metric")
        self.assertEqual(xgb_candidate._relation(get_metric("ROC AUC")), "proxy_not_host_metric")
        self.assertEqual(xgb_candidate._relation(get_metric("R2")), "monotone_equivalent_on_fixed_held")

    def test_unknown_resource_and_non_numeric_parameters_are_rejected(self):
        invalid = ({"device": "cuda"}, {"tree_method": "gpu_hist"}, {"nthread": 64},
                   {"objective": "reg:pseudohubererror"}, {"callbacks": []}, {"seed": 1},
                   {"max_depth": True}, {"learning_rate": True}, {"max_depth": 6.0},
                   {"max_depth": 0}, {"max_depth": 11}, {"max_bin": 4096},
                   {"learning_rate": float("nan")}, {"reg_lambda": float("inf")},
                   {"subsample": 0.1}, {"learning_rate": "0.1"}, {1: 2}, ["max_depth"])
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                recipe_for(parameters=parameters)

    def test_task_metric_policy_and_bounds_are_checked_before_training(self):
        cases = (lambda: xgb_candidate.make_xgboost_recipe("ranking", "RMSE", validation=TIME),
                 lambda: xgb_candidate.make_xgboost_recipe("classification", "RMSE", validation=TIME),
                 lambda: xgb_candidate.make_xgboost_recipe("regression", "RMSE", validation=None),
                 lambda: recipe_for("Macro F1", validation=TIME),
                 lambda: recipe_for("RMSE", validation={"strategy": "grouped"}),
                 lambda: recipe_for("RMSE", validation={"strategy": "auto"}),
                 lambda: recipe_for(feature_config={"unknown": True}))
        for index, build in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build()
        for iterations, stopping in ((49, 10), (2001, 10), (True, 10), (300.0, 10),
                                     (300, 0), (300, 201), (300, True), (50, 51)):
            with self.subTest(iterations=iterations, stopping=stopping), self.assertRaises(ValueError):
                recipe_for(max_iterations=iterations, stopping_rounds=stopping)

    def test_execution_rejects_tampered_or_unnormalized_recipes(self):
        training, target, prediction = data()
        recipe = recipe_for()
        tampered = []
        for key, value in (("runtime", dict(recipe["runtime"], device="cuda")),
                           ("loss", "absolute_error"), ("family", "xgboost_native_v2"),
                           ("version", True), ("preprocessing", "native_categorical")):
            tampered.append(dict(deepcopy(recipe), **{key: value}))
        tampered.append(dict(deepcopy(recipe), extra=1))
        for value in tampered:
            with self.subTest(value=value), self.assertRaises(ValueError):
                xgb_candidate.execute_xgboost_candidate(training, target, prediction, context(value))

    def test_context_must_match_recipe_identity(self):
        training, target, prediction = data()
        bad = context(recipe_for())
        bad["metric"] = get_metric("MAE").to_dict()
        with self.assertRaises(ValueError):
            xgb_candidate.execute_xgboost_candidate(training, target, prediction, bad)


class XGBoostEligibilityTests(unittest.TestCase):
    def test_expired_deadline_stops_before_any_fit(self):
        training, target, prediction = data()
        with recording() as calls, self.assertRaises(xgb_candidate.XGBoostDeadlineError) as caught:
            xgb_candidate.execute_xgboost_candidate(
                training, target, prediction, context(recipe_for(), deadline=time.monotonic() - 1))
        self.assertEqual(caught.exception.stage, "start")
        self.assertEqual(calls, {"feature_fit": [], "train": []})
        self.assertEqual(xgb_candidate.failure_evidence(caught.exception)["reason"], "deadline_exhausted")

    def test_resource_caps_skip_without_reducing_data(self):
        training, target, prediction = data()
        for limit, value, detail in (("MAX_FIT_ROWS", len(training) - 1, "fit_rows"),
                                     ("MAX_PREDICT_ROWS", 2, "predict_rows"),
                                     ("MAX_OUTPUT_FEATURES", 2, "output_features"),
                                     ("MAX_MATRIX_CELLS", 3 * len(training) - 1, "fit_matrix_cells"),
                                     ("MAX_CATEGORY_LEVELS", 3, "category_levels"),
                                     ("MAX_TOTAL_TREES", 199, "total_trees")):
            with self.subTest(limit=limit), patch.object(xgb_candidate, limit, value), recording() as calls:
                with self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
                    xgb_candidate.execute_xgboost_candidate(training, target, prediction,
                                                            context(recipe_for()))
                self.assertEqual((caught.exception.reason, caught.exception.detail),
                                 ("resource_cap_exceeded", detail))
                self.assertEqual(calls, {"feature_fit": [], "train": []})
                receipt = xgb_candidate.failure_evidence(caught.exception)
                self.assertEqual((receipt["status"], receipt["detail"]), ("skipped", detail))
                json.dumps(receipt, allow_nan=False)

    def test_missing_fit_class_and_unknown_labels_are_typed(self):
        training, target, prediction = data(classification=True)
        recipe = recipe_for("Macro F1")
        missing = target.replace("01", "NA")
        with self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
            xgb_candidate.execute_xgboost_candidate(training, missing, prediction,
                                                    context(recipe, classes=CLASSES))
        self.assertEqual(caught.exception.reason, "fit_missing_class")
        unknown = target.replace("01", "undeclared")
        with self.assertRaises(ValueError) as caught:
            xgb_candidate.execute_xgboost_candidate(training, unknown, prediction,
                                                    context(recipe, classes=CLASSES))
        self.assertNotIsInstance(caught.exception, xgb_candidate.XGBoostSkipError)

    def test_held_without_a_class_is_a_skip_not_a_random_replacement(self):
        training, target, prediction = data(classification=True)
        held = np.flatnonzero(target.to_numpy() != "01")[:20]
        fitting = np.setdiff1d(np.arange(len(training)), held)
        with patch.object(xgb_candidate, "split_rows", return_value=(fitting, held, {})), \
                recording() as calls, self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
            xgb_candidate.execute_xgboost_candidate(training, target, prediction,
                                                    context(recipe_for("Macro F1"), classes=CLASSES))
        self.assertEqual(caught.exception.reason, "internal_split_missing_class")
        self.assertEqual(calls["train"], [])

    def test_impossible_policies_are_skips(self):
        training, target, prediction = data(classification=True)
        same = training.assign(x="1", period="1", kind="same")
        with self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
            xgb_candidate.execute_xgboost_candidate(
                same, target, prediction,
                context(recipe_for("Macro F1", validation={"strategy": "grouped"}), classes=CLASSES))
        self.assertEqual(caught.exception.reason, "internal_split_unavailable")
        training, target, prediction = data()
        with self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
            xgb_candidate.execute_xgboost_candidate(training.assign(period="7"), target, prediction,
                                                    context(recipe_for()))
        self.assertEqual(caught.exception.reason, "internal_split_unavailable")

    def test_missing_or_other_xgboost_version_is_a_skip(self):
        training, target, prediction = data()
        with patch.dict("sys.modules", {"xgboost": None}), \
                self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
            xgb_candidate.execute_xgboost_candidate(training, target, prediction, context(recipe_for()))
        self.assertEqual(caught.exception.reason, "xgboost_unavailable")
        fake = type("FakeXGBoost", (), {"__version__": "3.0.0"})
        with patch.dict("sys.modules", {"xgboost": fake}), \
                self.assertRaises(xgb_candidate.XGBoostSkipError) as caught:
            xgb_candidate.execute_xgboost_candidate(training, target, prediction, context(recipe_for()))
        self.assertEqual(caught.exception.reason, "xgboost_version_mismatch")


class XGBoostGeneratedSourceTests(unittest.TestCase):
    def run_source(self, outcome):
        class Capture(StringIO):
            def __exit__(self, *args):
                self.flush()
                return False

        stream = Capture()
        with patch.object(xgb_candidate, "execute_xgboost_candidate", side_effect=outcome), \
                patch.object(Path, "open", autospec=True, return_value=stream) as writer:
            namespace = {}
            exec(compile(xgb_candidate.candidate_source(), "xgboost_generated.py", "exec"), namespace)
            try:
                value = namespace["fit_predict"](None, None, None, {})
            except Exception as error:  # noqa: BLE001 - the error is the observed result
                value = error
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(writer.call_args.args[0].name, xgb_candidate.EVIDENCE_NAME)
        self.assertEqual(writer.call_args.args[1], "x")
        return value, json.loads(stream.getvalue())

    def test_success_writes_one_exclusive_receipt_and_returns_strict_result(self):
        expected = {"predictions": [7.0, 8.0]}
        evidence = {"status": "completed", "fresh_full_refit": True}
        value, written = self.run_source(lambda *args, **kwargs: (expected, evidence))
        self.assertIs(value, expected)
        self.assertEqual(written, evidence)

    def test_skip_and_deadline_write_receipts_then_reraise(self):
        skip = xgb_candidate.XGBoostSkipError("resource_cap_exceeded", detail="fit_rows")
        late = xgb_candidate.XGBoostDeadlineError("refit_training")
        late.trace = {"trees": {"selected": 12}}
        for error, status, reason in ((skip, "skipped", "resource_cap_exceeded"),
                                      (late, "failed", "deadline_exhausted")):
            with self.subTest(status=status):
                value, written = self.run_source(error)
                self.assertIs(value, error)
                self.assertEqual((written["status"], written["reason"]), (status, reason))
        self.assertEqual(written["stage"], "refit_training")
        self.assertEqual(written["partial"]["trees"]["selected"], 12)

    def test_source_compiles(self):
        compile(xgb_candidate.candidate_source(), "xgboost_generated.py", "exec")


@unittest.skipUnless(NATIVE, "Optional xgboost is not installed")
class XGBoostNativeTests(unittest.TestCase):
    def invoke(self, metric="RMSE", *, classes=CLASSES, validation=None, prediction=None,
               phase="inner_selection", grouped=False, parameters=None):
        classification = get_metric(metric).classification
        training, target, default_prediction = data(classification=classification,
                                                    classes=classes, grouped=grouped)
        prediction = default_prediction if prediction is None else prediction
        recipe = recipe_for(metric, validation=validation, parameters=parameters)
        call_context = context(recipe, classes=classes if classification else (), phase=phase)
        before = deepcopy((training, target, prediction, call_context))
        with recording() as calls:
            result, evidence = xgb_candidate.execute_xgboost_candidate(
                training, target, prediction, call_context, return_evidence=True)
        after = (training, target, prediction, call_context)
        pd.testing.assert_frame_equal(after[0], before[0])
        pd.testing.assert_series_equal(after[1], before[1])
        pd.testing.assert_frame_equal(after[2], before[2])
        self.assertEqual(after[3], before[3])
        json.dumps(evidence, allow_nan=False)
        return result, evidence, calls, training, target

    def test_regression_time_policy_is_strictly_past_then_refits_all_rows(self):
        result, evidence, calls, training, target = self.invoke()
        fitting, held, _ = split_rows(training.reset_index(drop=True), target.reset_index(drop=True),
                                      FEATURES, False, TIME)
        self.assertEqual(len(calls["feature_fit"]), 2)
        internal, refit = calls["feature_fit"]
        self.assertEqual(internal.index.tolist(), fitting.tolist())
        self.assertLess(pd.to_numeric(internal["period"]).max(),
                        pd.to_numeric(training["period"].iloc[held]).min())
        self.assertEqual(set(training["period"].iloc[held]), {"4"})
        self.assertNotIn("held_only", set(internal["kind"]))
        self.assertNotIn("predict_only", set(internal["kind"]) | set(refit["kind"]))
        pd.testing.assert_frame_equal(refit, training.reset_index(drop=True))
        self.assertEqual(calls["train"], [{"rounds": 200, "early": True},
                                          {"rounds": evidence["trees"]["selected"], "early": False}])
        trees = evidence["trees"]
        self.assertEqual(trees["best"], trees["selected"])
        self.assertEqual(trees["refit_trained"], trees["selected"])
        self.assertLessEqual(trees["selected"], trees["internal_trained"])
        self.assertEqual((evidence["validation"]["strategy"], evidence["internal_validation_rows"],
                          evidence["full_fit_rows"], evidence["predict_rows"]),
                         ("time", len(held), len(training), 3))
        self.assertEqual(evidence["stopping"]["relation"], "same_quantity")
        self.assertNotEqual(evidence["internal_preprocessing_sha256"],
                            evidence["refit_preprocessing_sha256"])
        self.assertEqual(result["predictions"].shape, (3,))
        self.assertTrue(np.isfinite(result["predictions"]).all())
        self.assertLess(result["predictions"][0], result["predictions"][2])

    def test_grouped_policy_keeps_duplicate_feature_rows_disjoint(self):
        _, evidence, calls, training, _ = self.invoke("Macro F1", validation={"strategy": "grouped"},
                                                      grouped=True)
        internal = calls["feature_fit"][0]
        raw = training.reset_index(drop=True)
        held = raw.drop(index=internal.index)
        self.assertFalse(set(pd.util.hash_pandas_object(internal, index=False))
                         & set(pd.util.hash_pandas_object(held, index=False)))
        self.assertEqual(evidence["validation"]["group_overlap"], 0)

    def test_probability_columns_follow_literal_host_class_order(self):
        for classes in (CLASSES, list(reversed(CLASSES)), ["NA", "01"], ["01", "NA"]):
            with self.subTest(classes=classes):
                result, evidence, *_ = self.invoke("LogLoss", classes=classes)
                probabilities = result["predictions"]
                self.assertEqual(result["classes"], classes)
                self.assertEqual(probabilities.shape, (3, len(classes)))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
                # data() gives the lowest x the first literal class, the highest the last.
                self.assertEqual(classes[int(probabilities[0].argmax())], classes[0])
                self.assertEqual(classes[int(probabilities[2].argmax())], classes[-1])
                self.assertEqual(evidence["stopping"]["objective"],
                                 "binary:logistic" if len(classes) == 2 else "multi:softprob")

    def test_labels_are_argmax_of_the_same_aligned_probabilities(self):
        probabilities, *_ = self.invoke("LogLoss")
        labels, evidence, *_ = self.invoke("Macro F1")
        expected = np.asarray(CLASSES)[probabilities["predictions"].argmax(axis=1)]
        np.testing.assert_array_equal(labels["predictions"], expected)
        self.assertEqual(evidence["stopping"]["relation"], "proxy_not_host_metric")
        self.assertEqual(evidence["stopping"]["metric"], "mlogloss")

    def test_prediction_batch_and_phase_cannot_change_fit_or_tree_choice(self):
        other = pd.DataFrame({"x": ["1e6", "", "-1e6", "0"], "period": ["99"] * 4,
                              "kind": ["x", "y", "z", "NA"]}, index=list("abcd"))
        first = self.invoke()[1]
        changed = self.invoke(prediction=other)[1]
        refit = self.invoke(phase="full_refit")
        for evidence in (changed, refit[1]):
            for key in ("internal_preprocessing_sha256", "refit_preprocessing_sha256", "trees",
                        "validation", "internal_fit_rows"):
                self.assertEqual(evidence[key], first[key])
            self.assertEqual(evidence["stopping"]["internal_best_score"],
                             first["stopping"]["internal_best_score"])
        self.assertEqual(refit[1]["phase"], "full_refit")
        np.testing.assert_array_equal(refit[0]["predictions"], self.invoke()[0]["predictions"])

    def test_second_fixed_configuration_runs_with_absolute_error(self):
        result, evidence, *_ = self.invoke(
            "MAE", parameters=xgb_candidate.FIXED_CONFIGURATIONS["depth4_regularized_v1"])
        self.assertEqual((evidence["stopping"]["objective"], evidence["stopping"]["metric"]),
                         ("reg:absoluteerror", "mae"))
        self.assertEqual(evidence["recipe"]["parameters"]["max_depth"], 4)
        self.assertTrue(np.isfinite(result["predictions"]).all())

    def controlled_clock(self, on_train):
        clock = [1000.0]
        real_train = xgb_candidate._train
        calls = []

        def train(*args, **kwargs):
            calls.append(args[3])
            on_train(len(calls), clock, "before")
            result = real_train(*args, **kwargs)
            on_train(len(calls), clock, "after")
            return result

        training, target, prediction = data()
        with patch.object(xgb_candidate.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(xgb_candidate, "_train", side_effect=train), \
                self.assertRaises(xgb_candidate.XGBoostDeadlineError) as caught:
            xgb_candidate.execute_xgboost_candidate(training, target, prediction,
                                                    context(recipe_for(), deadline=1010.0))
        return caught.exception, calls

    def test_insufficient_measured_time_stops_before_refit(self):
        def after_internal(count, clock, moment):
            if (count, moment) == (1, "after"):
                clock[0] = 1009.9

        error, calls = self.controlled_clock(after_internal)
        self.assertEqual(error.stage, "refit_admission")
        self.assertEqual(error.trace["stopping"]["stopped_by"], "early_stopping")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("refit_trained", error.trace["trees"])

    def test_deadline_during_refit_is_a_failure_not_fewer_trees(self):
        def expire_in_refit(count, clock, moment):
            if (count, moment) == (2, "before"):
                clock[0] = 1011.0

        error, calls = self.controlled_clock(expire_in_refit)
        self.assertEqual(error.stage, "refit_training")
        self.assertEqual(len(calls), 2)
        receipt = xgb_candidate.failure_evidence(error)
        self.assertEqual((receipt["status"], receipt["stage"]), ("failed", "refit_training"))
        json.dumps(receipt, allow_nan=False)

    def test_direct_execution_has_no_file_io(self):
        with patch.object(Path, "write_text", side_effect=AssertionError("wrote a file")), \
                patch.object(Path, "open", side_effect=AssertionError("opened a file")):
            self.invoke()


def make_spec(root, metric, classes):
    metric_spec = get_metric(metric)
    if metric_spec.prediction_kind == "probabilities":
        outputs = ("row_key", *classes)
        probability_columns = tuple((label, label) for label in classes)
    else:
        outputs, probability_columns = ("row_key", "answer"), ()
    return TaskSpec(root / "original_task.md", root / "original_private_data", "row_key", "answer",
                    tuple(FEATURES), ("x", "period"), ("kind",),
                    tuple(classes) if metric_spec.classification else (), outputs,
                    probability_columns, classes[-1] if metric_spec.classification else None,
                    metric_spec)


@unittest.skipUnless(NATIVE, "Optional xgboost is not installed")
class XGBoostSubprocessTests(unittest.TestCase):
    def frames(self, metric, classes=CLASSES):
        training, target, _ = data(classification=get_metric(metric).classification, classes=classes)
        frame = training.reset_index(drop=True)
        frame.insert(0, "row_key", [f"r{index:04d}" for index in range(len(frame))])
        frame["answer"] = [str(value) for value in target]
        held = frame.index % 5 == 0
        return frame.loc[~held].reset_index(drop=True), frame.loc[held].reset_index(drop=True)

    def test_native_protocol_receipt_rescore_and_refit(self):
        for metric, classes in (("ROC AUC", ["NA", "01"]), ("Macro F1", CLASSES)):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec = make_spec(root, metric, classes)
                fit, holdout = self.frames(metric, classes)
                actuals = holdout.pop("answer").tolist()
                source = root / "candidate.py"
                source.write_text(xgb_candidate.candidate_source(), encoding="utf-8")
                frozen = freeze_candidate(source, recipe_for(metric), root / "frozen")
                result = run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=holdout,
                                       output_dir=root / "runs", phase="outer_validation",
                                       timeout=300, actuals=actuals)
                saved = pd.read_csv(result.output_path, dtype=str, keep_default_na=False)
                self.assertEqual(list(saved.columns), list(spec.output_columns))
                self.assertEqual(saved["row_key"].tolist(), holdout["row_key"].tolist())
                if spec.prediction_kind == "probabilities":
                    values = saved.loc[:, list(classes)].to_numpy(dtype=float)
                    self.assertEqual(values.shape, (len(holdout), len(classes)))
                    rescore = score_predictions(spec.metric, actuals, values, classes=classes,
                                                prediction_classes=classes,
                                                positive_class=spec.positive_class)
                else:
                    rescore = score_predictions(spec.metric, actuals, saved["answer"].to_numpy(),
                                                classes=classes)
                self.assertEqual(rescore, result.metric["value"])
                evidence = json.loads((result.output_path.parent / xgb_candidate.EVIDENCE_NAME)
                                      .read_text(encoding="utf-8"))
                self.assertEqual((evidence["status"], evidence["phase"], evidence["predict_rows"]),
                                 ("completed", "outer_validation", len(holdout)))
                receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["phase"], "outer_validation")
                full = pd.concat([fit, holdout.assign(answer=actuals)], ignore_index=True)
                refit = run_candidate(frozen, spec=spec, fit_frame=full, predict_frame=holdout,
                                      output_dir=root / "runs", phase="full_refit", timeout=300)
                self.assertIsNone(refit.metric)
                refit_evidence = json.loads((refit.output_path.parent / xgb_candidate.EVIDENCE_NAME)
                                            .read_text(encoding="utf-8"))
                self.assertEqual(refit_evidence["full_fit_rows"], len(full))
                self.assertEqual(refit.recipe_hash, result.recipe_hash)

    def test_native_skip_writes_receipt_and_fails_the_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = make_spec(root, "Macro F1", CLASSES)
            fit, holdout = self.frames("Macro F1")
            fit = fit.loc[fit["answer"].ne("01")].reset_index(drop=True)
            holdout = holdout.drop(columns="answer")
            source = root / "candidate.py"
            source.write_text(xgb_candidate.candidate_source(), encoding="utf-8")
            frozen = freeze_candidate(source, recipe_for("Macro F1"), root / "frozen")
            with self.assertRaises(CandidateExecutionError) as caught:
                run_candidate(frozen, spec=spec, fit_frame=fit, predict_frame=holdout,
                              output_dir=root / "runs", phase="inner_selection", timeout=300)
            self.assertNotEqual(caught.exception.execution["exit_code"], 0)
            receipts = list((root / "runs").rglob(xgb_candidate.EVIDENCE_NAME))
            self.assertEqual(len(receipts), 1)
            evidence = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertEqual((evidence["status"], evidence["reason"]), ("skipped", "fit_missing_class"))
            self.assertFalse(list((root / "runs").rglob("predictions.csv")))


if __name__ == "__main__":
    unittest.main()
