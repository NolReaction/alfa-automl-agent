"""Explicit, retained real-model smoke from the organizer's flat package.

Ordinary unit discovery does not start unrecorded model fits. Set
ROUND6_NATIVE_SMOKE_OUTPUT to a new external directory for the one authorized
run. The retained summary counts each model fit; this toy is not quality data.
"""
from __future__ import annotations

import hashlib
import json
import math
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
from agent import round6_regression_candidate as candidate
from agent.round6_regression_base import active_component
from agent.metrics import get_metric
from catboost import CatBoostRegressor

assert agent.project_root() == root
assert Path(candidate.__file__).resolve().is_relative_to(root)
assert not (root / 'agent').exists()
rows = [{'period': str(period), 'value': str(key / 10), 'kind': str(key % 3)}
        for period in (1, 2, 3) for key in range(500)]
X = pd.DataFrame(rows, index=[1000 + i * 7 for i in range(len(rows))])
y = pd.Series([3 * float(row['value']) + 20 * float(row['period'])
               + (int(row['kind']) - 1) ** 2 for row in rows], index=X.index)
query = pd.DataFrame({'period': ['4', '4', '4'], 'value': ['2.6', '0.2', '3.4'],
                      'kind': ['unseen', '1', '2']}, index=[91, 17, 8])
extended = pd.concat([query.iloc[[2, 0, 1]], pd.DataFrame({
    'period': ['9999'], 'value': ['1000000'], 'kind': ['query_only']})])
metric = get_metric('RMSE')
recipe = candidate.make_round6_recipe('period', 'S20', history_config=None)
source = candidate.candidate_source()
(root / 'frozen_candidate.py').write_text(source, encoding='utf-8')
(root / 'frozen_recipe.json').write_text(json.dumps(recipe, sort_keys=True, indent=2), encoding='utf-8')
original = (X.copy(deep=True), y.copy(deep=True), query.copy(deep=True), deepcopy(recipe))
results = []
fit_events = []
analytic_events = []
current_directory = None
def instrument(original_fit, engine):
    def measured(model, frame, target, *args, **kwargs):
        metadata = active_component() or {}
        assert metadata.get('engine', engine) == engine
        phase = metadata.get('phase') or ('full_refit' if len(frame) == len(X) else 'incumbent_internal_selection')
        ordinal = 1 + sum(row['event'] == 'started' for row in fit_events)
        started = time.monotonic()
        def record(event, **extra):
            row = {'event': event, 'fit_ordinal': ordinal, 'engine': engine,
                'rows': len(frame), 'phase': phase,
                'component': metadata.get('id', 'catboost_all'),
                'deadline_monotonic': metadata.get('deadline_monotonic', current_deadline),
                'original_deadline_monotonic': metadata.get('original_deadline_monotonic', current_deadline),
                'path': metadata.get('path', 'correction'),
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
original_analytic_fit = candidate.fit_correction
def measured_analytic_fit(arm, target, predictions):
    metadata = active_component() or {}
    assert metadata.get('engine') == 'analytic_residual'
    ordinal = 1 + sum(row['event'] == 'started' for row in analytic_events)
    started = time.monotonic()
    def record(event, **extra):
        row = {'event': event, 'fit_ordinal': ordinal, 'engine': 'analytic_residual',
            'rows': len(target), 'phase': 'correction_fit_A', 'kind': 'analytic_fit',
            'component': metadata.get('id', 'residual_correction'),
            'deadline_monotonic': metadata.get('deadline_monotonic', current_deadline),
            'original_deadline_monotonic': metadata.get('original_deadline_monotonic', current_deadline),
            'path': metadata.get('path', 'analytic_correction'),
            'monotonic': time.monotonic(), **extra}
        analytic_events.append(row)
        with (current_directory / 'analytic_fit_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
    record('started')
    try:
        result = original_analytic_fit(arm, target, predictions)
    except BaseException as error:
        record('failed', exception_type=type(error).__name__, elapsed_seconds=time.monotonic() - started)
        raise
    record('completed', elapsed_seconds=time.monotonic() - started)
    return result
candidate.fit_correction = measured_analytic_fit
for index, prediction_frame in enumerate((query, extended)):
    directory = root / ('invocation_' + str(index + 1))
    directory.mkdir()
    current_directory = directory
    event_start = len(fit_events)
    analytic_start = len(analytic_events)
    import os
    os.chdir(directory)
    current_deadline = time.monotonic() + 90
    context = {'features': list(X), 'numeric': ['period', 'value'], 'categorical': ['kind'],
        'task_type': 'regression', 'prediction_kind': metric.prediction_kind,
        'metric': metric.to_dict(), 'recipe': deepcopy(recipe),
        'deadline_monotonic': current_deadline}
    before = deepcopy(context)
    namespace = {}
    exec(compile(source, str(root / 'frozen_candidate.py'), 'exec'), namespace)
    started = time.monotonic()
    fit_frame = X if index == 0 else X.iloc[::-1].reset_index(drop=True)
    fit_target = y if index == 0 else y.iloc[::-1].reset_index(drop=True)
    result = namespace['fit_predict'](fit_frame, fit_target, prediction_frame, context)
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
    assert len(started_fits) == len(completed_fits)
    assert len(started_fits) in (3, 4), 'Toy must exercise correction selection and aligned recheck'
    assert [event['phase'] for event in started_fits[:2]] == [
        'correction_selection_A', 'prefix_recheck']
    assert all(event['engine'] == 'catboost' for event in started_fits)
    assert all(event['rows'] == 1000 for event in started_fits[:2])
    assert started_fits[-1]['rows'] == len(X)
    assert set(pd.read_csv(directory / 'regression_ablation.csv')) >= {'identity', 'base', 'calibrated'}
    assert all(event['rows'] == len(X) for event in completed_fits if event['phase'] == 'full_refit')
    analytic = analytic_events[analytic_start:]
    analytic_started = [event for event in analytic if event['event'] == 'started']
    analytic_completed = [event for event in analytic if event['event'] == 'completed']
    assert len(analytic_started) == len(analytic_completed) == 1
    assert analytic_started[0]['rows'] == evidence['correction']['split']['A_rows']
    assert evidence['correction']['analytic_parameter_fit_calls'] == len(analytic_started)
    assert all(event['phase'] == 'correction_fit_A' and event['engine'] == 'analytic_residual' for event in analytic)
    np.savetxt(directory / 'predictions.csv', values, delimiter=',', header='prediction', comments='')
    results.append({'predictions': values.tolist(), 'evidence': evidence, 'elapsed_seconds': elapsed,
        'actual_started_fits': len(started_fits), 'actual_completed_fits': len(completed_fits),
        'physical_fit_count': len(started_fits), 'completed_fit_count': len(completed_fits),
        'actual_analytic_fit_count': len(analytic_started),
        'completed_analytic_fit_count': len(analytic_completed),
        'total_fit_operations': len(started_fits) + len(analytic_started),
        'actual_fit_events': events, 'actual_analytic_fit_events': analytic})
np.testing.assert_allclose(np.asarray(results[0]['predictions']),
    np.asarray(results[1]['predictions'])[[1, 2, 0]], rtol=1e-10, atol=1e-10)
pd.testing.assert_frame_equal(X, original[0])
pd.testing.assert_series_equal(y, original[1])
pd.testing.assert_frame_equal(query, original[2])
assert recipe == original[3]
summary = {'kind': 'real_flat_package_toy_smoke_not_quality_comparison',
    'source_checkout_imported': False, 'api_requests': 0, 'thread_limit': 8,
    'arm': 'S20', 'invocations': results, 'query_batch_invariance': True,
    'training_row_order_invariance': True,
    'physical_fit_count': sum(row['physical_fit_count'] for row in results),
    'completed_fit_count': sum(row['completed_fit_count'] for row in results),
    'actual_analytic_fit_count': sum(row['actual_analytic_fit_count'] for row in results),
    'completed_analytic_fit_count': sum(row['completed_analytic_fit_count'] for row in results),
    'total_fit_operations': sum(row['total_fit_operations'] for row in results),
    'inputs_unchanged': True, 'invocation_cap_seconds': 90,
    'calibration_gate_is_fit_only': True,
    'rows_fit': len(X), 'query_rows': [len(query), len(extended)],
    'environment': {'python': sys.version.split()[0], 'packages': {
        name: importlib.metadata.version(name) for name in ('numpy','pandas','catboost')}}}
(root / 'flat_smoke_summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'completed': True, 'invocations': len(results), 'query_batch_invariance': True}))
'''


def run_flat_smoke(output: Path):
    """Run two serial, capped invocations and preserve evidence even on failure."""
    started = time.monotonic()
    original_deadline = started + 195.0
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "flat_package.zip"
    package = output / "package"
    stdout, stderr, returncode, stop_reason = "", "", None, "smoke_preparation_failed"
    preparation_seconds, child_timeout_seconds, error_type = None, None, None
    try:
        names = build_archive(archive)
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
        preparation_seconds = time.monotonic() - started
        child_timeout_seconds = original_deadline - time.monotonic()
        if not math.isfinite(child_timeout_seconds) or child_timeout_seconds <= 0:
            stop_reason = "smoke_preparation_deadline_exceeded"
            raise TimeoutError("Flat smoke preparation exhausted the original allowance")
        result = subprocess.run([sys.executable, "-I", "-c", PROBE, str(package)],
            cwd=output, env=environment, capture_output=True, text=True, timeout=child_timeout_seconds)
        stdout, stderr, returncode, stop_reason = result.stdout, result.stderr, result.returncode, "completed_process"
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        stdout, stderr, returncode, stop_reason = decoded(error.stdout), decoded(error.stderr), None, "child_timeout"
        error_type = type(error).__name__
    except Exception as error:
        stderr, error_type = str(error), type(error).__name__
    (output / "smoke_stdout.log").write_text(stdout, encoding="utf-8")
    (output / "smoke_stderr.log").write_text(stderr, encoding="utf-8")
    engine_events, analytic_events = [], []
    for invocation in sorted(package.glob("invocation_*")):
        for filename, events in (("native_fit_events.jsonl", engine_events),
                                 ("analytic_fit_events.jsonl", analytic_events)):
            path = invocation / filename
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        events.append(json.loads(line))
    physical_started = sum(row["event"] == "started" for row in engine_events)
    physical_completed = sum(row["event"] == "completed" for row in engine_events)
    analytic_started = sum(row["event"] == "started" for row in analytic_events)
    analytic_completed = sum(row["event"] == "completed" for row in analytic_events)

    def digest_if_present(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    receipt = {"returncode": returncode, "stop_reason": stop_reason, "error_type": error_type,
               "started_monotonic": started, "original_deadline_monotonic": original_deadline,
               "cap_seconds": 195.0, "preparation_seconds": preparation_seconds,
               "child_timeout_seconds": child_timeout_seconds,
               "physical_fit_count": physical_started, "completed_fit_count": physical_completed,
               "actual_analytic_fit_count": analytic_started, "completed_analytic_fit_count": analytic_completed,
               "total_fit_operations": physical_started + analytic_started,
               "archive_sha256": digest_if_present(archive),
               "probe_sha256": hashlib.sha256(PROBE.encode()).hexdigest(),
               "packaged_dependency_hashes_sha256": digest_if_present(output / "packaged_dependency_hashes.json"),
               "scope": "toy_portability_and_contract_only"}
    elapsed = time.monotonic() - started
    receipt["elapsed_seconds"] = elapsed if math.isfinite(elapsed) and elapsed >= 0 else None
    if receipt["elapsed_seconds"] is None or elapsed > 195.0:
        receipt.update(returncode=None, stop_reason="original_smoke_deadline_or_clock_failure")
    (output / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    if receipt["returncode"] != 0:
        raise AssertionError("Real flat smoke failed (" + receipt["stop_reason"] + "); retained stdout/stderr: " + stderr[-4000:])
    return json.loads((package / "flat_smoke_summary.json").read_text())


class Round6NativeSmoke(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ROUND6_NATIVE_SMOKE_OUTPUT"),
                         "Explicit output directory required for accounted native fits")
    def test_flat_package_real_models_and_query_batch_invariance(self):
        summary = run_flat_smoke(Path(os.environ["ROUND6_NATIVE_SMOKE_OUTPUT"]))
        self.assertEqual(len(summary["invocations"]), 2)
        self.assertTrue(summary["query_batch_invariance"])


if __name__ == "__main__":
    unittest.main()
