"""A bounded target-prior checkpoint for failures before useful model output.

This is structurally checked, full-train prior output, not a validated ML winner.
The immutable receipt and CSV hashes are retained by the host process. Separate
files under the same OS account are not a security sandbox.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import time
import tempfile
from uuid import uuid4

from agent import project_root
from agent.automl import INPUT_NAMES, _atomic_json, _plain
from agent.execution import run_python
from agent.llm import safe_error_message
from offline_baseline import csv_header, find_data, read_csv


CHUNK_ROWS = 10000
MAX_PREPARE_SECONDS = 30.0
MAX_PREPARE_FRACTION = 0.05
MAX_WORKER_RECEIPT_BYTES = 1024 * 1024


def _before_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Emergency checkpoint preparation budget exhausted")


def _hash_file(path, deadline=None):
    digest = hashlib.sha256()
    with _plain(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _before_deadline(deadline)
            digest.update(block)
    _before_deadline(deadline)
    return digest.hexdigest()


def _bounded_copy(source, destination, deadline=None):
    from agent.file_io import replace_file

    source, destination = _plain(source), _plain(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".emergency-output-",
                                     suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)
        try:
            with source.open("rb") as input_stream:
                for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                    _before_deadline(deadline)
                    output.write(block)
        except BaseException:
            output.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        replace_file(temporary, destination, check=lambda: (
            _plain(temporary), _plain(destination), _before_deadline(deadline)))
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(task_path, data_dir, deadline=None):
    result = {"task.md": _hash_file(task_path, deadline)}
    for name in INPUT_NAMES:
        result[name] = _hash_file(data_dir / name, deadline)
    return result


def _prior_spec(task_path, data_dir, metric, classes, headers, metadata):
    """Use TaskSpec's exact output contract without loading train feature data."""
    from agent.metrics import infer_positive_class
    from agent.task_spec import (
        TaskSpec, _auc_declaration, _class_list, _declaration, _positive_class,
    )
    import re

    text = task_path.read_text(encoding="utf-8-sig")
    roles = metadata["role"].str.strip().str.lower()
    id_fields = metadata.loc[roles == "id", "feature"].tolist()
    targets = metadata.loc[roles == "target", "feature"].tolist()
    if len(id_fields) != 1 or len(targets) != 1:
        raise ValueError("Emergency prior requires one declared ID and target")
    id_column, target = id_fields[0], targets[0]
    dtypes = dict(zip(metadata["feature"], metadata["dtype"].str.strip().str.lower()))
    role_map = dict(zip(metadata["feature"], roles))
    if dtypes[target] not in {"numeric", "category"} and not (
            metric.classification and dtypes[target] == "binary"):
        raise ValueError("Unsupported target dtype for emergency prior")
    train_header, test_header, columns = headers
    if id_column not in train_header or target not in train_header or id_column not in test_header:
        raise ValueError("Declared ID or target missing from input header")
    if target in test_header:
        raise ValueError("Original test input contains the target")
    features = tuple(name for name in train_header if name not in {id_column, target})
    if not features or set(features) != set(test_header) - {id_column}:
        raise ValueError("Train/test emergency feature headers do not match")
    for name in features:
        if role_map.get(name) != "feature" or dtypes.get(name) not in {"numeric", "category", "binary"}:
            raise ValueError("Invalid emergency feature dictionary")
    numeric = tuple(name for name in features if dtypes[name] in {"numeric", "binary"})
    categorical = tuple(name for name in features if dtypes[name] == "category")
    order = _class_list(text)
    if order is not None:
        if not metric.classification or set(order) != set(classes):
            raise ValueError("Class order does not match training labels")
        classes = order
    declared_type = _declaration(text, r"task[\s_-]*type|тип\s+задачи")
    if declared_type is not None:
        is_class = bool(re.search(r"classif|классифика", declared_type, re.IGNORECASE))
        is_reg = bool(re.search(r"regress|регресси", declared_type, re.IGNORECASE))
        if is_class == is_reg or metric.classification != is_class:
            raise ValueError("Declared task type conflicts with emergency metric")
    positive = _positive_class(text)
    if positive is not None and positive not in classes:
        raise ValueError("Positive class does not match training labels")
    probabilities = ()
    if metric.prediction_kind == "probabilities":
        output = [column for column in columns if column != id_column]
        if len(output) == 1:
            if len(classes) != 2:
                raise ValueError("Single probability output requires binary classes")
            inferred = output[0] if output[0] in classes and output[0] != target else infer_positive_class(classes)
            positive = positive if positive is not None else inferred
            if positive is None or (output[0] in classes and output[0] != positive):
                raise ValueError("Ambiguous emergency positive probability class")
            probabilities = ((output[0], positive),)
        elif len(output) == len(classes) and set(output) == set(classes):
            probabilities = tuple((column, column) for column in output)
            if len(classes) == 2 and positive is None:
                positive = infer_positive_class(classes) or classes[-1]
        else:
            raise ValueError("Emergency probability columns do not cover classes")
        if metric.name == "ROC AUC" and len(classes) > 2:
            declaration = _auc_declaration(text)
            if (not (re.search(r"\bovr\b|one[ -]vs[ -]rest", declaration, re.IGNORECASE)
                     and re.search(r"\bmacro\b", declaration, re.IGNORECASE))
                    or re.search(r"\b(?:ovo|micro|weighted)\b|one[ -]vs[ -]one", declaration, re.IGNORECASE)):
                raise ValueError("Multiclass AUC requires explicit macro OVR")
    return TaskSpec(task_path, data_dir, id_column, target, features, numeric, categorical,
                    tuple(classes), tuple(columns), probabilities, positive, metric)


def _fit_prior(task_path, directory, nonce):
    """Trusted child worker: fit target statistics, then predict original test IDs."""
    import numpy as np
    import pandas as pd
    from agent.task_contract import parse_task_metric

    task_path, directory = Path(task_path), Path(directory)
    data_dir = find_data(task_path.parent)
    metric = parse_task_metric(task_path.read_text(encoding="utf-8-sig"))
    metadata = read_csv(data_dir / "data_dictionary.csv")
    if (not {"feature", "role", "dtype"} <= set(metadata.columns)
            or metadata["feature"].duplicated().any()):
        raise ValueError("Invalid emergency data dictionary")
    roles = metadata["role"].str.strip().str.lower()
    targets = metadata.loc[roles == "target", "feature"].tolist()
    id_fields = metadata.loc[roles == "id", "feature"].tolist()
    if len(targets) != 1 or len(id_fields) != 1:
        raise ValueError("Emergency target or ID is ambiguous")
    target, id_column = targets[0], id_fields[0]
    headers = [csv_header(data_dir / name) for name in ("train.csv", "test.csv", "sample_submission.csv")]
    counts, values, fit_rows = Counter(), [], 0
    weighted_means, seen_training_ids = [], set()
    with pd.read_csv(data_dir / "train.csv", usecols=[id_column, target], dtype=str,
                     keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_ROWS) as training:
        for chunk in training:
            ids = chunk[id_column].tolist()
            if (any(not value.strip() or value in seen_training_ids for value in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Emergency training IDs are empty or duplicated")
            seen_training_ids.update(ids)
            series = chunk[target]
            if series.str.strip().eq("").any():
                raise ValueError("Emergency training target contains missing values")
            fit_rows += len(series)
            if metric.classification:
                counts.update(series.tolist())
            else:
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(numeric).all():
                    raise ValueError("Emergency training target must be finite numeric values")
                if metric.name == "MAE":
                    values.append(numeric)
                else:
                    scale = float(np.max(np.abs(numeric))) if len(numeric) else 0.0
                    mean = float(np.mean(numeric / scale)) * scale if scale else 0.0
                    weighted_means.append((mean, len(numeric)))
    if fit_rows < 2:
        raise ValueError("Emergency prior requires at least two training targets")
    classes = tuple(sorted(counts)) if metric.classification else ()
    spec = _prior_spec(task_path, data_dir, metric, classes, headers, metadata)
    if metric.classification:
        prediction = max(spec.classes, key=lambda label: counts[label])
        priors = [counts[label] / fit_rows for label in spec.classes]
        model = {"strategy": "class_prior", "fit_rows": fit_rows,
                 "classes": list(spec.classes), "probabilities": priors, "label": prediction}
    else:
        if metric.name == "MAE":
            targets = np.concatenate(values)
            middle = len(targets) // 2
            part = np.partition(targets, (middle - 1, middle) if len(targets) % 2 == 0 else middle)
            prediction = (float(part[middle - 1]) / 2 + float(part[middle]) / 2
                          if len(targets) % 2 == 0 else float(part[middle]))
        else:
            prediction = math.fsum(mean * (rows / fit_rows) for mean, rows in weighted_means)
        if not math.isfinite(prediction):
            raise ValueError("Emergency target prior is not finite")
        model = {"strategy": "median" if metric.name == "MAE" else "mean",
                 "fit_rows": fit_rows, "value": prediction}
    output = directory / "prior.csv"
    first, seen_ids, predict_rows = True, set(), 0
    with pd.read_csv(data_dir / "test.csv", usecols=[spec.id_column], dtype=str,
                     keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_ROWS) as testing:
        for chunk in testing:
            ids = chunk[spec.id_column].tolist()
            if any(not value.strip() or value in seen_ids for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("Emergency test IDs are empty or duplicated")
            seen_ids.update(ids)
            predict_rows += len(ids)
            if spec.prediction_kind == "probabilities":
                predicted = np.tile(priors, (len(ids), 1))
                frame = spec.prediction_frame(ids, predicted, prediction_classes=spec.classes)
            else:
                frame = spec.prediction_frame(ids, [prediction] * len(ids))
            frame.to_csv(output, index=False, mode="w" if first else "a", header=first, encoding="utf-8")
            first = False
    if predict_rows == 0:
        raise ValueError("Emergency test is empty")
    _atomic_json(directory / "worker.json", {"nonce": nonce, "spec": spec.to_dict(),
                                           "model": model, "predict_rows": predict_rows})


class EmergencyCheckpoint:
    """One host-owned, hash-bound prior, never an arbitrary existing CSV."""

    def __init__(self, task_path: Path, directory: Path | None = None):
        self.task_path = _plain(Path(task_path)).resolve()
        self.task_dir = self.task_path.parent
        self.directory = (_plain(Path(directory)).resolve() if directory is not None else
                          self.task_dir / ".agent_runs" / uuid4().hex)
        if not self.directory.is_relative_to(self.task_dir / ".agent_runs"):
            raise ValueError("Emergency directory must be inside task/.agent_runs")
        self.report_path = self.directory / "emergency.json"
        self._metadata = {"status": "not_started", "kind": "target_prior", "quality_verified": False,
                          "report_path": str(self.report_path)}
        self._data_dir = None
        self._inputs = None
        self._spec = None
        self._csv_hash = None
        self._receipt_hash = None

    @property
    def metadata(self):
        return dict(self._metadata)

    @property
    def inputs_frozen(self):
        return self._inputs is not None and self._data_dir is not None

    def check_inputs(self, *, deadline=None):
        if self._inputs is None or self._data_dir is None:
            raise ValueError("Emergency input fingerprint is unavailable")
        if (_plain(find_data(self.task_dir)).resolve() != self._data_dir
                or _fingerprint(self.task_path, self._data_dir, deadline) != self._inputs):
            raise ValueError("Original task inputs changed; emergency publication cancelled")

    def _validate_csv(self, path, *, deadline=None):
        import pandas as pd
        if csv_header(path) != list(self._spec.output_columns):
            raise ValueError("Emergency CSV header differs from the task output contract")
        rows = 0
        with pd.read_csv(_plain(path), dtype=str, keep_default_na=False,
                         encoding="utf-8-sig", chunksize=CHUNK_ROWS) as predictions, \
                pd.read_csv(self._data_dir / "test.csv", usecols=[self._spec.id_column], dtype=str,
                            keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_ROWS) as ids:
            for predicted, expected in zip_longest(predictions, ids):
                _before_deadline(deadline)
                if predicted is None or expected is None:
                    raise ValueError("Emergency prediction count differs from test rows")
                self._spec.validate_frame(predicted, expected[self._spec.id_column].tolist())
                rows += len(predicted)
        if rows == 0:
            raise ValueError("Emergency predictions are empty")

    def prepare(self, timeout=30.0, *, deadline=None):
        if (type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0
                or (deadline is not None and (type(deadline) not in {int, float} or not math.isfinite(deadline)))):
            raise ValueError("Emergency timeout/deadline must be finite")
        if self._receipt_hash is not None:
            raise ValueError("Emergency checkpoint has already been prepared")
        started = time.monotonic()
        remaining = max(0.0, deadline - started) if deadline is not None else timeout / MAX_PREPARE_FRACTION
        preparation_deadline = min(started + min(timeout, MAX_PREPARE_SECONDS,
                                                remaining * MAX_PREPARE_FRACTION),
                                   deadline if deadline is not None else math.inf)
        self._metadata.update(status="preparing")
        try:
            _before_deadline(preparation_deadline)
            self._data_dir = _plain(find_data(self.task_dir)).resolve()
            self._inputs = _fingerprint(self.task_path, self._data_dir, preparation_deadline)
            _plain(self.directory)
            self.directory.mkdir(parents=True, exist_ok=False)
            nonce = uuid4().hex
            code = ("import sys\nfrom pathlib import Path\n"
                    f"sys.path.insert(0, {str(project_root())!r})\n"
                    "from agent.emergency import _fit_prior\n"
                    f"_fit_prior(Path({str(self.task_path)!r}), Path({str(self.directory)!r}), {nonce!r})\n")
            remaining = preparation_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Emergency preparation budget exhausted before worker launch")
            execution = run_python(code, self.task_dir, self.directory / "execution", 0,
                                   timeout=remaining, max_memory_bytes=2 * 1024 ** 3)
            if (execution.get("exit_code") != 0 or execution.get("error")
                    or execution.get("timed_out") or execution.get("memory_limit_exceeded")):
                raise RuntimeError("Emergency prior worker failed or exhausted its bounded budget")
            _before_deadline(preparation_deadline)
            worker_path = _plain(self.directory / "worker.json")
            if worker_path.stat().st_size > MAX_WORKER_RECEIPT_BYTES:
                raise ValueError("Emergency worker receipt exceeds its bounded metadata limit")
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            if worker.get("nonce") != nonce:
                raise ValueError("Emergency worker receipt is stale")
            from agent.task_spec import TaskSpec
            self._spec = TaskSpec.from_dict(worker["spec"])
            if self._spec.task_path != self.task_path or self._spec.data_dir != self._data_dir:
                raise ValueError("Emergency worker task paths differ from frozen input paths")
            prior = _plain(self.directory / "prior.csv")
            self._validate_csv(prior, deadline=preparation_deadline)
            self.check_inputs(deadline=preparation_deadline)
            csv_hash = _hash_file(prior, preparation_deadline)
            receipt = {"version": 1, "kind": "target_prior", "quality_verified": False,
                       "status": "ready", "nonce": nonce, "inputs": self._inputs,
                       "submission_sha256": csv_hash, "model": worker["model"],
                       "predict_rows": worker["predict_rows"], "spec": worker["spec"]}
            _atomic_json(self.report_path, receipt)
            receipt_hash = _hash_file(self.report_path, preparation_deadline)
            # Keep both expected hashes in host memory. Rewriting the CSV and its
            # public receipt together is not valid recovery evidence.
            self._csv_hash, self._receipt_hash = csv_hash, receipt_hash
            destination = _plain(self.task_dir / "submission.csv")
            existing_valid = False
            if destination.is_file():
                try:
                    self._validate_csv(destination, deadline=preparation_deadline)
                    existing_valid = True
                except (OSError, ValueError):
                    pass
            _before_deadline(preparation_deadline)
            if not existing_valid:
                self.check_inputs(deadline=preparation_deadline)
                _bounded_copy(prior, destination, preparation_deadline)
            self._metadata.update(status="ready", available=True, published=not existing_valid,
                                  preserved_existing=existing_valid, model=worker["model"],
                                  elapsed_seconds=round(time.monotonic() - started, 6))
            return True
        except Exception as error:
            self._csv_hash = self._receipt_hash = None
            self._metadata.update(status="unavailable", available=False,
                                  error=safe_error_message(error),
                                  elapsed_seconds=round(time.monotonic() - started, 6))
            return False

    def restore(self, *, deadline=None):
        if self._receipt_hash is None or self._csv_hash is None:
            return False
        try:
            self.check_inputs(deadline=deadline)
            prior = _plain(self.directory / "prior.csv")
            receipt = _plain(self.report_path)
            if _hash_file(prior, deadline) != self._csv_hash or _hash_file(receipt, deadline) != self._receipt_hash:
                raise ValueError("Emergency checkpoint bytes or receipt changed")
            self._validate_csv(prior, deadline=deadline)
            self.check_inputs(deadline=deadline)
            _bounded_copy(prior, _plain(self.task_dir / "submission.csv"), deadline)
            self._metadata.update(status="restored", restored=True)
            return True
        except Exception as error:
            self._metadata.update(status="recovery_rejected", restored=False, error=safe_error_message(error))
            return False
