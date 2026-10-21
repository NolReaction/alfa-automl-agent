"""Regressions for strict CSV records, literal features and real-valued scores."""

from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from agent.features import FeatureTransformer, MISSING_CATEGORY, RARE_CATEGORY
from agent.metrics import label_values, probability_matrix, score_predictions
from offline_baseline import make_pipeline, prepare_features, read_csv


class StrictRecordTests(unittest.TestCase):
    def test_overwide_record_cannot_silently_become_dataframe_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            original = "id,x,target\n001,1,a,EXTRA\n002,2,b,OTHER\n"
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"record 2 has 4 fields; expected 3"):
                read_csv(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_omitted_trailing_field_is_not_a_valid_explicit_empty_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.csv"
            path.write_text("id,x,kind\n001,1\n002,2,a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"record 2 has 2 fields; expected 3"):
                read_csv(path)

    def test_blank_record_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.csv"
            path.write_text("id,x\n001,1\n\n002,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"record 3 has 0 fields; expected 2"):
                read_csv(path)

    def test_quoted_commas_newlines_bom_and_explicit_empty_cells_remain_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.csv"
            path.write_bytes(
                '\ufeffid,category,target\r\n001,"a,b",NA\r\n'
                'NA,"line one\nline two",01\r\n003,,"class c"\r\n'.encode("utf-8")
            )
            result = read_csv(path)
            self.assertEqual(result.index.tolist(), [0, 1, 2])
            self.assertEqual(result["id"].tolist(), ["001", "NA", "003"])
            self.assertEqual(result["category"].tolist(), ["a,b", "line one\nline two", ""])
            self.assertEqual(result["target"].tolist(), ["NA", "01", "class c"])


class RealNumericEvidenceTests(unittest.TestCase):
    def invalid_matrices(self):
        return (
            np.asarray([[0, 1]], dtype="datetime64[D]"),
            np.asarray([[0, 1]], dtype="timedelta64[D]"),
            np.asarray([[.25 + .8j, .75 - .8j]]),
            np.asarray([[False, True]]),
            np.asarray([[date(1970, 1, 1), 1]], dtype=object),
            np.asarray([[datetime(1970, 1, 1), 1]], dtype=object),
            np.asarray([[timedelta(0), 1]], dtype=object),
            np.asarray([[np.datetime64("1970-01-01"), 1]], dtype=object),
            np.asarray([[np.timedelta64(0, "D"), 1]], dtype=object),
            np.asarray([[.25 + .8j, .75]], dtype=object),
            np.asarray([[False, 1]], dtype=object),
        )

    def test_probability_evidence_rejects_lossy_numeric_casts(self):
        for matrix in self.invalid_matrices():
            with self.subTest(dtype=str(matrix.dtype), values=repr(matrix)):
                with self.assertRaises(ValueError):
                    probability_matrix(matrix, classes=("01", "NA"))

    def test_binary_probability_vectors_reject_lossy_casts_too(self):
        for vector in (np.asarray([0, 1], dtype="datetime64[D]"),
                       np.asarray([0, 1], dtype="timedelta64[D]"),
                       np.asarray([.1 + 3j, .9 - 3j]), np.asarray([False, True])):
            with self.subTest(dtype=str(vector.dtype)), self.assertRaises(ValueError):
                probability_matrix(vector, classes=("no", "yes"), positive_class="yes")

    def test_mixed_python_bool_probabilities_are_checked_before_numpy_promotion(self):
        for values in ([[True, 0.0]], [[False, 1.0]], [[np.bool_(True), 0]],
                       [True, .2], [[True], [.2]]):
            with self.subTest(values=repr(values)), self.assertRaises(ValueError):
                probability_matrix(values, classes=("no", "yes"), positive_class="yes")
        # Numbers already represented by a typed float array remain valid.
        np.testing.assert_array_equal(
            probability_matrix(np.array([[True, 0.0]], dtype=float), classes=("no", "yes")),
            [[1., 0.]],
        )

    def test_regression_scores_reject_nonreal_predictions_and_actuals(self):
        vectors = (np.asarray([1, 2], dtype="datetime64[D]"),
                   np.asarray([1, 2], dtype="timedelta64[D]"),
                   np.asarray([1 + 500j, 2 - 900j]), np.asarray([False, True]))
        for metric in ("RMSE", "MSE", "MAE", "R2"):
            for vector in vectors:
                with self.subTest(metric=metric, dtype=str(vector.dtype)):
                    with self.assertRaises(ValueError):
                        score_predictions(metric, [1, 2], vector)
                    with self.assertRaises(ValueError):
                        score_predictions(metric, vector, [1, 2])

    def test_mixed_python_bool_regression_values_are_checked_before_vector_coercion(self):
        for metric in ("RMSE", "MSE", "MAE", "R2"):
            for values in ([True, 2.], [1, False], [[True], [2.]], [np.bool_(False), 2]):
                with self.subTest(metric=metric, values=repr(values)):
                    with self.assertRaises(ValueError):
                        score_predictions(metric, [1., 2.], values)
                    with self.assertRaises(ValueError):
                        score_predictions(metric, values, [1., 2.])
        self.assertEqual(score_predictions("RMSE", [1., 2.], np.array([True, 2.], dtype=float)), 0.)

    def test_numeric_csv_strings_and_probability_column_permutations_still_work(self):
        matrix = probability_matrix(
            [["0.8", "0.2"], ["0.1", "0.9"]],
            classes=("NA", "01"), prediction_classes=("01", "NA"),
        )
        np.testing.assert_allclose(matrix, [[.2, .8], [.9, .1]])
        self.assertEqual(score_predictions("RMSE", ["-2", "3.5"], ["-2.0", "3.50"]), 0.)
        self.assertEqual(score_predictions("ROC AUC", ["NA", "01"], ["0.9", "0.1"],
                                           classes=("01", "NA"), positive_class="NA"), 1.)

    def test_genuine_missing_label_scalars_do_not_become_literal_class_names(self):
        for value in (None, pd.NA, pd.NaT, np.datetime64("NaT"), np.timedelta64("NaT")):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "missing or empty"):
                    label_values([value, "yes"])
                with self.assertRaises(ValueError):
                    score_predictions("Accuracy", [value, "yes"], ["<NA>", "yes"],
                                      classes=("<NA>", "NaT", "yes"))

    def test_literal_missing_marker_strings_remain_exact_labels(self):
        labels = ["NA", "NaT", "<NA>", "nan", "01", " class c ", "класс с пробелом"]
        np.testing.assert_array_equal(label_values(labels), labels)
        self.assertEqual(score_predictions("Accuracy", labels, labels, classes=labels), 1.)


class LiteralFeatureTests(unittest.TestCase):
    def test_legacy_missing_token_and_escape_namespace_cannot_collide(self):
        prefix = "__ALFA_LEGACY_VALUE__"
        raw = pd.DataFrame({"kind": ["", "__MISSING__", prefix + "__MISSING__",
                                     prefix, "NA", " "]})
        before = raw.copy(deep=True)
        prepared = prepare_features(raw, ["kind"], [], ["kind"])
        self.assertTrue(pd.isna(prepared["kind"].iloc[0]))
        self.assertEqual(prepared["kind"].iloc[1:].tolist(),
                         [prefix + "__MISSING__", prefix * 2 + "__MISSING__",
                          prefix * 2, "NA", " "])
        preprocessing = make_pipeline([], ["kind"], True).named_steps["prepare"]
        encoded = preprocessing.fit_transform(prepared)
        self.assertEqual(len(np.unique(encoded)), len(raw))
        # Each independent raw batch follows the same deterministic mapping.
        independent = prepare_features(raw.iloc[::-1], ["kind"], [], ["kind"])
        np.testing.assert_array_equal(preprocessing.transform(independent), encoded[::-1])
        pd.testing.assert_frame_equal(raw, before)

    def test_all_empty_categorical_column_remains_valid_for_legacy_preprocessing(self):
        raw = pd.DataFrame({"empty_num": [""] * 12, "constant": ["2"] * 12,
                            "empty_category": [""] * 12})
        before = raw.copy(deep=True)
        prepared = prepare_features(raw, list(raw), ["empty_num", "constant"], ["empty_category"])
        self.assertEqual(prepared["empty_category"].dtype, object)
        self.assertTrue(prepared["empty_category"].isna().all())
        # Fit the actual production baseline's preprocessing, not a replacement.
        preprocessing = make_pipeline(["empty_num", "constant"], ["empty_category"], True).named_steps["prepare"]
        fitted = preprocessing.fit_transform(prepared)
        self.assertEqual(fitted.shape, (12, 3))
        self.assertTrue(np.isfinite(fitted).all())
        held = raw.iloc[:2].copy()
        held["empty_category"] = ["unseen", ""]
        inferred = preprocessing.transform(prepare_features(
            held, list(raw), ["empty_num", "constant"], ["empty_category"]))
        self.assertEqual(inferred.shape, (2, 3))
        self.assertNotEqual(inferred[0, 2], inferred[1, 2])
        pd.testing.assert_frame_equal(raw, before)

    def test_whitespace_categories_are_not_empty_missing_cells(self):
        frame = pd.DataFrame({"категория пробел": ["", " ", "\t", "\u3000", "NA", None]})
        transformed = FeatureTransformer([], ["категория пробел"]).fit_transform(frame)
        self.assertEqual(transformed.iloc[:, 0].tolist(),
                         [MISSING_CATEGORY, " ", "\t", "\u3000", "NA", MISSING_CATEGORY])
        self.assertEqual(transformed.iloc[:, 0].nunique(), 5)

    def test_whitespace_category_counts_are_learned_on_fit_only(self):
        train = pd.DataFrame({"kind": [" ", " ", "NA", "NA", ""]})
        feature = FeatureTransformer([], ["kind"], {"rare_min_count": 2}).fit(train)
        self.assertEqual(feature.allowed_categories_["kind"], frozenset({" ", "NA"}))
        held = pd.DataFrame({"kind": [" ", "unseen", ""]})
        self.assertEqual(feature.transform(held)["kind"].tolist(),
                         [" ", RARE_CATEGORY, MISSING_CATEGORY])
        self.assertEqual(feature.allowed_categories_["kind"], frozenset({" ", "NA"}))

    def test_whitespace_text_preserves_length_and_is_not_marked_missing(self):
        frame = pd.DataFrame({"text": ["   ", "", None, "NA"]})
        transformed = FeatureTransformer([], ["text"], {"text_columns": ["text"]}).fit_transform(frame)
        self.assertEqual(transformed["text__length"].tolist(), [3., 0., 0., 2.])
        self.assertEqual(transformed["text__word_count"].tolist(), [0., 0., 0., 1.])
        self.assertEqual(transformed["text__missing"].tolist(), [0., 1., 1., 0.])


if __name__ == "__main__":
    unittest.main()
