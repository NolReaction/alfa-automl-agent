"""Explicit, retained real-model smoke from the organizer's flat package.

Ordinary unit discovery does not start unrecorded model fits. Set
ROUND5_NATIVE_SMOKE_OUTPUT to a new external directory for the one authorized
run. The retained summary counts each model fit; this toy is not quality data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
import zipfile

from build_submission import build_archive


PROBE = r'''
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
import agent
from agent import round5_regression_candidate as candidate
from agent.round5_regression_base import active_component
from agent.metrics import get_metric
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

assert agent.project_root() == root
assert Path(candidate.__file__).resolve().is_relative_to(root)
assert not (root / 'agent').exists()
rows = [{'period': str(period), 'value': str(key / 5), 'kind': str(key % 3)}
        for period in (1, 2, 3) for key in range(20)]
X = pd.DataFrame(rows, index=[1000 + i * 7 for i in range(len(rows))])
y = pd.Series([2 * float(row['value']) + .3 * float(row['period'])
               + (int(row['kind']) - 1) ** 2 for row in rows], index=X.index)
query = pd.DataFrame({'period': ['4', '4', '4'], 'value': ['2.6', '0.2', '3.4'],
                      'kind': ['unseen', '1', '2']}, index=[91, 17, 8])
extended = pd.concat([query.iloc[[2, 0, 1]], pd.DataFrame({
    'period': ['9999'], 'value': ['1000000'], 'kind': ['query_only']})])
metric = get_metric('RMSE')
recipe = candidate.make_round5_recipe('period', 'E50', history_config=None)
source = candidate.candidate_source()
(root / 'frozen_candidate.py').write_text(source, encoding='utf-8')
(root / 'frozen_recipe.json').write_text(json.dumps(recipe, sort_keys=True, indent=2), encoding='utf-8')
original = (X.copy(deep=True), y.copy(deep=True), query.copy(deep=True), deepcopy(recipe))
results = []
fit_events = []
current_directory = None
def instrument(original_fit, engine):
    def measured(model, frame, target, *args, **kwargs):
        metadata = active_component()
        assert metadata is not None and metadata['engine'] == engine
        ordinal = 1 + sum(row['event'] == 'started' for row in fit_events)
        started = time.monotonic()
        def record(event, **extra):
            row = {'event': event, 'fit_ordinal': ordinal, 'engine': engine,
                'rows': len(frame), 'phase': metadata['phase'],
                'component': metadata['id'], 'deadline_monotonic': metadata['deadline_monotonic'],
                'monotonic': time.monotonic(), **extra}
            fit_events.append(row)
            with (current_directory / 'native_fit_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
        record('started')
        try:
            result = original_fit(model, frame, target, *args, **kwargs)
        except BaseException as error:
            record('failed', exception_type=type(error).__name__, elapsed_seconds=time.monotonic() - started)
            raise
        record('completed', elapsed_seconds=time.monotonic() - started)
        return result
    return measured
CatBoostRegressor.fit = instrument(CatBoostRegressor.fit, 'catboost')
LGBMRegressor.fit = instrument(LGBMRegressor.fit, 'lightgbm')
for index, prediction_frame in enumerate((query, extended)):
    directory = root / ('invocation_' + str(index + 1))
    directory.mkdir()
    current_directory = directory
    event_start = len(fit_events)
    import os
    os.chdir(directory)
    context = {'features': list(X), 'numeric': ['period', 'value'], 'categorical': ['kind'],
        'task_type': 'regression', 'prediction_kind': metric.prediction_kind,
        'metric': metric.to_dict(), 'recipe': deepcopy(recipe),
        'deadline_monotonic': time.monotonic() + 90}
    before = deepcopy(context)
    namespace = {}
    exec(compile(source, str(root / 'frozen_candidate.py'), 'exec'), namespace)
    started = time.monotonic()
    result = namespace['fit_predict'](X, y, prediction_frame, context)
    elapsed = time.monotonic() - started
    assert set(result) == {'predictions'}
    values = np.asarray(result['predictions'], dtype=float)
    assert values.shape == (len(prediction_frame),) and np.isfinite(values).all()
    assert context == before and elapsed < 90
    evidence = json.loads((directory / 'regression_evidence.json').read_text())
    assert evidence['status'] == 'completed'
    assert evidence['history']['config'] is None
    assert evidence['full_fit_rows'] == len(X)
    assert evidence['fresh_full_refit'] is True
    assert evidence['ablation_sha256'] == hashlib.sha256((directory / 'regression_ablation.csv').read_bytes()).hexdigest()
    events = fit_events[event_start:]
    started_fits = [event for event in events if event['event'] == 'started']
    completed_fits = [event for event in events if event['event'] == 'completed']
    assert len(started_fits) == len(completed_fits) == 4
    assert [(event['engine'], event['phase']) for event in started_fits] == [
        ('catboost', 'internal_selection'), ('lightgbm', 'internal_selection'),
        ('catboost', 'full_refit'), ('lightgbm', 'full_refit')]
    assert all(event['rows'] == len(X) for event in completed_fits if event['phase'] == 'full_refit')
    np.savetxt(directory / 'predictions.csv', values, delimiter=',', header='prediction', comments='')
    results.append({'predictions': values.tolist(), 'evidence': evidence, 'elapsed_seconds': elapsed,
        'actual_started_fits': len(started_fits), 'actual_completed_fits': len(completed_fits),
        'actual_fit_events': events})
np.testing.assert_allclose(np.asarray(results[0]['predictions']),
    np.asarray(results[1]['predictions'])[[1, 2, 0]], rtol=1e-10, atol=1e-10)
pd.testing.assert_frame_equal(X, original[0])
pd.testing.assert_series_equal(y, original[1])
pd.testing.assert_frame_equal(query, original[2])
assert recipe == original[3]
summary = {'kind': 'real_flat_package_toy_smoke_not_quality_comparison',
    'source_checkout_imported': False, 'api_requests': 0, 'thread_limit': 8,
    'arm': 'E50', 'invocations': results, 'query_batch_invariance': True,
    'inputs_unchanged': True, 'invocation_cap_seconds': 90,
    'rows_fit': len(X), 'query_rows': [len(query), len(extended)],
    'environment': {'python': sys.version.split()[0], 'packages': {
        name: importlib.metadata.version(name) for name in ('numpy','pandas','catboost','lightgbm')}}}
(root / 'flat_smoke_summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'completed': True, 'invocations': len(results), 'query_batch_invariance': True}))
'''


def run_flat_smoke(output: Path):
    """Run two serial, capped invocations and preserve evidence even on failure."""
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "flat_package.zip"
    names = build_archive(archive)
    package = output / "package"
    package.mkdir()
    with zipfile.ZipFile(archive) as stream:
        for name in names:
            if "/" in name or not (name.endswith(".py") or name == "requirements.txt"):
                raise AssertionError("Flat archive has an unsupported entry")
            (package / name).write_bytes(stream.read(name))
    if (package / "agent").exists():
        raise AssertionError("Smoke cannot rely on an agent directory")
    (output / "smoke_probe.py").write_text(PROBE, encoding="utf-8")
    packaged_hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}
    (output / "packaged_dependency_hashes.json").write_text(
        json.dumps(packaged_hashes, sort_keys=True, indent=2), encoding="utf-8")
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"PYTHONPATH", "PYTHONHOME"} and not name.startswith("OPENAI_")}
    environment.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, "-I", "-c", PROBE, str(package)],
            cwd=output, env=environment, capture_output=True, text=True, timeout=195)
        stdout, stderr, returncode, stop_reason = result.stdout, result.stderr, result.returncode, "completed_process"
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        stdout, stderr, returncode, stop_reason = decoded(error.stdout), decoded(error.stderr), None, "child_timeout"
    (output / "smoke_stdout.log").write_text(stdout, encoding="utf-8")
    (output / "smoke_stderr.log").write_text(stderr, encoding="utf-8")
    receipt = {"returncode": returncode, "stop_reason": stop_reason, "elapsed_seconds": time.monotonic() - started,
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "probe_sha256": hashlib.sha256(PROBE.encode()).hexdigest(),
               "packaged_dependency_hashes_sha256": hashlib.sha256(
                   (output / "packaged_dependency_hashes.json").read_bytes()).hexdigest(),
               "scope": "toy_portability_and_contract_only"}
    (output / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if returncode != 0:
        raise AssertionError("Real flat smoke failed (" + stop_reason + "); retained stdout/stderr: " + stderr[-4000:])
    return json.loads((package / "flat_smoke_summary.json").read_text())


class Round5NativeSmoke(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ROUND5_NATIVE_SMOKE_OUTPUT"),
                         "Explicit output directory required for accounted native fits")
    def test_flat_package_real_models_and_query_batch_invariance(self):
        summary = run_flat_smoke(Path(os.environ["ROUND5_NATIVE_SMOKE_OUTPUT"]))
        self.assertEqual(len(summary["invocations"]), 2)
        self.assertTrue(summary["query_batch_invariance"])


if __name__ == "__main__":
    unittest.main()
