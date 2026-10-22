import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment_rescue_classification_rescore import audit, macro_f1


class PortableEvidenceTests(unittest.TestCase):
    def test_macro_f1_includes_declared_missing_class(self):
        score = macro_f1(['a', 'a', 'b'], ['a', 'b', 'b'], ['a', 'b', 'c'])
        self.assertAlmostEqual(score['macro_f1'], 4 / 9)
        self.assertEqual(score['per_class']['c']['support'], 0)

    def test_export_rescore_rejects_tamper_and_reordered_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = 'id,target\nx,a\ny,b\n'
            (root / 'p.csv').write_text(content)
            (root / 'a.csv').write_text(content)
            manifest = {'id_column': 'id', 'target_column': 'target', 'classes': ['a', 'b'],
                        'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ['p.csv', 'a.csv']},
                        'scored_predictions': [{'predictions': 'p.csv', 'answers': 'a.csv',
                                                 'candidate': 'test', 'seed': 42, 'macro_f1': 1.}]}
            (root / 'evidence_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(audit(root)['results'][0]['macro_f1'], 1.)
            (root / 'p.csv').write_text('id,target\ny,b\nx,a\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                audit(root)
            manifest['files']['p.csv'] = hashlib.sha256((root / 'p.csv').read_bytes()).hexdigest()
            (root / 'evidence_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'IDs differ'):
                audit(root)

    def test_unknown_labels_rejected(self):
        with self.assertRaises(ValueError):
            macro_f1(['a'], ['alien'], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
