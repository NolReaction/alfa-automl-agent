"""Versioned composition risks; model fitting is exercised by serial experiments."""
from copy import deepcopy
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from agent import round4_regression_candidate as candidate
from agent.regression_candidate import make_regression_recipe, _recipe as validate_v1


def fixture():
    X = pd.DataFrame({'period': ['1', '1', '2', '2'], 'entity': ['a', 'b', 'a', 'b'],
                      'value': ['2', '3', '4', '5']})
    y = pd.Series([2., 3., 4., 5.])
    predict = pd.DataFrame({'period': ['3', '3'], 'entity': ['a', 'cold'], 'value': ['6', '7']})
    config = {'key_columns': ['entity'], 'period_column': 'period', 'numeric_columns': [],
              'positive_columns': [], 'ambiguity_policy': 'past_unique_only'}
    recipe = candidate.make_round4_recipe('period', history_config=config,
                                          features={'category_frequencies': True})
    context = {'features': list(X), 'numeric': ['period', 'value'], 'categorical': ['entity'],
               'recipe': recipe, 'deadline_monotonic': time.monotonic() + 60}
    return X, y, predict, context


class CandidateContractTests(unittest.TestCase):
    def test_new_family_does_not_reinterpret_normalized_v1(self):
        old = make_regression_recipe('period')
        self.assertEqual(validate_v1(old), old)
        with self.assertRaises(ValueError):
            candidate._recipe(old)
        new = candidate.make_round4_recipe('period')
        with self.assertRaises(ValueError):
            validate_v1(new)
        self.assertEqual(new['base'], old['base'])
        self.assertEqual(candidate._recipe(json.loads(json.dumps(new))), new)

    def test_recipe_is_detached_and_rejects_hidden_controls(self):
        *_, context = fixture()
        original = deepcopy(context['recipe'])
        checked = candidate._recipe(context['recipe'])
        checked['features']['category_frequencies'] = False
        self.assertEqual(context['recipe'], original)
        for field, value in [('features', {}), ('calibration_alignment', 'unknown'),
                             ('version', True), ('family', 'rescue_regression_v1')]:
            bad = deepcopy(original); bad[field] = value
            with self.assertRaises(ValueError):
                candidate._recipe(bad)
        bad = deepcopy(original); bad['deadline_monotonic'] = 99999999
        with self.assertRaises(ValueError):
            candidate._recipe(bad)

    def test_base_receives_raw_rows_and_history_uses_only_past(self):
        X, y, predict, context = fixture()
        snapshots = [item.copy(deep=True) for item in (X, y, predict)]
        before = deepcopy(context)
        seen = []
        def base(fit, target, query, ctx, recipe, **kwargs):
            pd.testing.assert_frame_equal(fit, X)
            pd.testing.assert_frame_equal(query, predict)
            self.assertLessEqual(ctx['deadline_monotonic'], context['deadline_monotonic'])
            seen.append(kwargs)
            return np.array([10., 10.]), {'physical_fits': 2}, {
                'fit_positions': np.array([0, 1]), 'validation_positions': np.array([2, 3]),
                'predictions': np.array([10., 10.])}
        with patch.object(candidate, 'fit_base', side_effect=base):
            result, evidence, alternatives = candidate.execute_round4_candidate(
                X, y, predict, context, return_evidence=True)
        # Only earlier y=2,4 contributes mean3 to the known object, cold stays10.
        np.testing.assert_array_equal(result['predictions'], [6.5, 10.])
        self.assertEqual(evidence['history']['reason'], 'insufficient_calibration_history_mean_fallback')
        self.assertEqual(seen[0]['feature_config'], context['recipe']['features'])
        self.assertEqual(context, before)
        for actual, expected in zip((X, y, predict), snapshots):
            (pd.testing.assert_series_equal if isinstance(actual, pd.Series)
             else pd.testing.assert_frame_equal)(actual, expected)

    def test_declared_roles_reject_invalid_schema_before_model_call(self):
        X, y, predict, context = fixture()
        context['features'].remove('entity')
        with patch.object(candidate, 'fit_base') as base:
            with self.assertRaises(ValueError):
                candidate.fit_predict(X, y, predict, context)
            base.assert_not_called()

    def test_source_imports_versioned_adapter_and_keeps_standard_contract(self):
        source = candidate.candidate_source()
        self.assertIn('from agent.round4_regression_candidate import execute_round4_candidate', source)
        namespace = {}
        exec(compile(source, '<round4-candidate>', 'exec'), namespace)
        self.assertTrue(callable(namespace['fit_predict']))


if __name__ == '__main__':
    unittest.main()
