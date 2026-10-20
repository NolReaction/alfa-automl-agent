"""Frozen, subprocess-only recipes for research v3.

Generated code defines ``fit_predict(X_fit, y_fit, X_predict, context)`` and
returns ``{"predictions": values, "classes": class_order}``.  ``classes`` is
mandatory for probabilities.  Targets used for host scoring never enter the
child's input files.  Integrity checks protect the ordinary protocol; this is
not an operating-system sandbox for arbitrary Python running as the same user.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from agent import project_root
from agent.execution import run_python
from agent.file_io import replace_file
from agent.metrics import score_predictions
from agent.resources import DEFAULT_MEMORY_LIMIT_BYTES
from agent.task_spec import TaskSpec
from offline_baseline import read_csv


PROTOCOL_VERSION = 3
CANDIDATE_NAME = "research_candidate.py"
RECIPE_NAME = "research_recipe.json"
MAX_CODE_BYTES = 256 * 1024
MAX_RECIPE_BYTES = 64 * 1024
PHASES = {"selection", "final_check", "refit", "inner_selection", "outer_validation", "full_refit"}


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_bytes(path: Path, *, limit: int | None = None) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Candidate input must be a regular file: {path.name}")
    if limit is not None and path.stat().st_size > limit:
        raise ValueError(f"Candidate file exceeds its size limit: {path.name}")
    value = path.read_bytes()
    if limit is not None and len(value) > limit:
        raise ValueError(f"Candidate file exceeds its size limit: {path.name}")
    return value


def _json_bytes(value: dict) -> bytes:
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Recipe object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("Recipe must contain only JSON values")
    if not isinstance(value, dict):
        raise ValueError("Recipe must be a JSON object")
    try:
        check(value)
        result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Recipe must contain finite, serializable JSON values") from error
    if len(result) > MAX_RECIPE_BYTES:
        raise ValueError("Recipe exceeds 64 KiB")
    return result


@dataclass(frozen=True)
class FrozenCandidate:
    code_path: Path
    recipe_path: Path
    code_hash: str
    recipe_hash: str

    def verify(self) -> None:
        if (_hash(_regular_bytes(self.code_path, limit=MAX_CODE_BYTES)) != self.code_hash
                or _hash(_regular_bytes(self.recipe_path, limit=MAX_RECIPE_BYTES)) != self.recipe_hash):
            raise ValueError("Frozen candidate code or recipe changed")

    @property
    def recipe(self) -> dict:
        self.verify()
        return json.loads(self.recipe_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CandidateResult:
    output_path: Path
    metric: dict | None
    execution: dict
    code_hash: str
    recipe_hash: str
    phase: str
    receipt_path: Path
    output_hash: str
    output_size: int
    receipt_hash: str

    def read_output(self) -> bytes:
        """Read the accepted bytes once, rejecting later cross-stage edits."""
        output = _regular_bytes(self.output_path, limit=self.output_size)
        receipt = _regular_bytes(self.receipt_path, limit=16 * 1024)
        if (len(output) != self.output_size or _hash(output) != self.output_hash
                or _hash(receipt) != self.receipt_hash):
            raise ValueError("Verified candidate output or receipt changed")
        return output

    def verify(self) -> None:
        self.read_output()

    def copy_output(self, destination: Path) -> None:
        """Atomically copy the pinned bytes, rather than rereading a mutable path."""
        output = self.read_output()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".candidate-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(output)
                stream.flush()
                os.fsync(stream.fileno())
            replace_file(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class CandidateExecutionError(ValueError):
    def __init__(self, message: str, execution: dict):
        super().__init__(message)
        self.execution = execution


def freeze_candidate(source_path: Path, recipe: dict, destination: Path) -> FrozenCandidate:
    """Snapshot code and recipe once; reading this function never imports code."""
    code = _regular_bytes(Path(source_path), limit=MAX_CODE_BYTES)
    if not code.strip():
        raise ValueError("Candidate source must not be empty")
    # A syntax check executes nothing and is not a security filter.
    try:
        compile(code, CANDIDATE_NAME, "exec")
    except (SyntaxError, ValueError) as error:
        raise ValueError("Candidate source is not valid Python") from error
    recipe_bytes = _json_bytes(recipe)
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    code_path, recipe_path = destination / CANDIDATE_NAME, destination / RECIPE_NAME
    code_path.write_bytes(code)
    recipe_path.write_bytes(recipe_bytes)
    candidate = FrozenCandidate(code_path, recipe_path, _hash(code), _hash(recipe_bytes))
    (destination / "frozen_candidate.json").write_bytes(_json_bytes({
        "version": PROTOCOL_VERSION, "code_hash": candidate.code_hash,
        "recipe_hash": candidate.recipe_hash,
    }))
    candidate.verify()
    return candidate


def _check_frames(spec: TaskSpec, fit_frame, predict_frame) -> None:
    for frame, label in ((fit_frame, "fit"), (predict_frame, "predict")):
        if not len(frame) or frame.columns.duplicated().any():
            raise ValueError(f"Candidate {label} data must have rows and unique columns")
        required = {spec.id_column, *spec.features}
        if label == "fit":
            required.add(spec.target_column)
        elif spec.target_column in frame.columns:
            raise ValueError("Prediction inputs must not contain target answers")
        if not required <= set(frame.columns):
            raise ValueError(f"Candidate {label} data is missing required columns")
        ids = frame[spec.id_column].tolist()
        if (any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Candidate row IDs must be unique non-empty literal strings")


def _output_limit(spec: TaskSpec, ids: list[str]) -> int:
    # Bound the host's read before CSV parsing.  Normal numeric spellings fit in
    # 128 bytes; literal labels and IDs receive their measured escaped lengths.
    prediction_bytes = (2 * max(len(label.encode("utf-8")) for label in spec.classes) + 16
                        if spec.prediction_kind == "labels"
                        else 128 * len(spec.prediction_columns))
    header = sum(2 * len(column.encode("utf-8")) + 4 for column in spec.output_columns)
    return max(64 * 1024, header + sum(
        2 * len(value.encode("utf-8")) + prediction_bytes + 32 for value in ids))


def _driver(root: Path) -> str:
    # The generated module is loaded only here, inside the bounded child.
    # The context's TaskSpec paths describe this invocation, not the source task.
    return f'''import hashlib, json, runpy, sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
import pandas as pd
from agent.task_spec import TaskSpec
context = json.loads(Path("context.json").read_text(encoding="utf-8"))
spec = TaskSpec.from_dict(context["task_spec"])
fit = pd.read_csv("fit.csv", dtype=str, keep_default_na=False)
predict = pd.read_csv("predict.csv", dtype=str, keep_default_na=False)
y = fit[spec.target_column].copy()
if not spec.classification:
    y = pd.to_numeric(y, errors="raise")
module = runpy.run_path("candidate.py", run_name="_alfa_research_candidate_")
function = module.get("fit_predict")
if not callable(function):
    raise ValueError("Candidate must define callable fit_predict")
public_context = {{key: value for key, value in context.items() if key != "task_spec"}}
value = function(fit.loc[:, list(spec.features)].copy(), y,
                 predict.loc[:, list(spec.features)].copy(), public_context)
if not isinstance(value, dict) or "predictions" not in value or set(value) - {{"predictions", "classes"}}:
    raise ValueError("fit_predict must return predictions and optional classes")
classes = value.get("classes")
if spec.prediction_kind == "probabilities":
    if (not isinstance(classes, (list, tuple)) or not classes
            or any(not isinstance(label, str) for label in classes)):
        raise ValueError("Probability predictions require an explicit literal class order")
frame = spec.prediction_frame(predict[spec.id_column].tolist(), value["predictions"],
                              prediction_classes=classes)
temporary = Path(".predictions.csv.tmp")
frame.to_csv(temporary, index=False)
temporary.replace("predictions.csv")
receipt = {{"version": {PROTOCOL_VERSION}, "nonce": context["nonce"],
           "phase": context["phase"], "code_hash": context["code_hash"],
           "recipe_hash": context["recipe_hash"],
           "output_hash": hashlib.sha256(Path("predictions.csv").read_bytes()).hexdigest()}}
Path("receipt.json").write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")
'''


def run_candidate(candidate: FrozenCandidate, *, spec: TaskSpec, fit_frame,
                  predict_frame, output_dir: Path, phase: str, timeout: float,
                  actuals=None,
                  max_memory_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES) -> CandidateResult:
    """Fit a fixed recipe and validate predictions, keeping evaluation labels host-only.

    Every invocation has fresh inputs and outputs.  ``actuals`` may be supplied
    for selection/final_check; refit never accepts validation answers.  The
    caller owns split selection and must stop the research dialogue before
    final_check.  Phase naming by itself cannot enforce the caller's lifecycle.
    """
    if not isinstance(phase, str) or phase not in PHASES:
        raise ValueError("Unsupported candidate phase")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Candidate timeout must be positive and finite")
    if phase in {"refit", "full_refit"} and actuals is not None:
        raise ValueError("Refit does not take evaluation answers")
    candidate.verify()
    _check_frames(spec, fit_frame, predict_frame)
    expected_ids = predict_frame[spec.id_column].tolist()
    frozen_actuals = list(actuals) if actuals is not None else None
    if frozen_actuals is not None and len(frozen_actuals) != len(expected_ids):
        raise ValueError("Frozen evaluation answers do not match prediction rows")
    invocation = Path(output_dir).absolute() / (phase + "_" + uuid4().hex)
    invocation.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(candidate.code_path, invocation / "candidate.py")
    shutil.copyfile(candidate.recipe_path, invocation / "recipe.json")
    fit_frame.loc[:, [spec.id_column, *spec.features, spec.target_column]].to_csv(
        invocation / "fit.csv", index=False)
    predict_frame.loc[:, [spec.id_column, *spec.features]].to_csv(
        invocation / "predict.csv", index=False)
    local_spec = spec.to_dict()
    local_spec.update(task_path=str(invocation / "task.md"), data_dir=str(invocation))
    nonce = uuid4().hex
    context = {
        "version": PROTOCOL_VERSION, "protocol_version": PROTOCOL_VERSION,
        "phase": phase, "nonce": nonce, "recipe": candidate.recipe,
        "code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash,
        "features": list(spec.features), "numeric": list(spec.numeric),
        "categorical": list(spec.categorical), "classes": list(spec.classes),
        "positive_class": spec.positive_class, "metric": spec.metric.to_dict(),
        "task_type": spec.metric.task_type, "prediction_kind": spec.prediction_kind,
        "task_spec": local_spec,
    }
    # Context can exceed the recipe limit when a task has many feature names.
    (invocation / "context.json").write_text(json.dumps(
        context, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    inputs = {name: _hash(_regular_bytes(invocation / name)) for name in
              ("candidate.py", "recipe.json", "fit.csv", "predict.csv", "context.json")}
    if inputs["candidate.py"] != candidate.code_hash or inputs["recipe.json"] != candidate.recipe_hash:
        raise ValueError("Frozen candidate changed while copying inputs")
    execution = run_python(_driver(project_root()), invocation, invocation / "logs", 0,
                           timeout=timeout, max_memory_bytes=max_memory_bytes)
    candidate.verify()
    if any(_hash(_regular_bytes(invocation / name)) != digest for name, digest in inputs.items()):
        raise CandidateExecutionError("Candidate changed its execution inputs", execution)
    if ("error" in execution or execution.get("exit_code") != 0
            or execution.get("timed_out") or execution.get("memory_limit_exceeded")):
        raise CandidateExecutionError("Candidate subprocess did not complete successfully", execution)
    output_path, receipt_path = invocation / "predictions.csv", invocation / "receipt.json"
    try:
        output = _regular_bytes(output_path, limit=_output_limit(spec, expected_ids))
        receipt_bytes = _regular_bytes(receipt_path, limit=16 * 1024)
        receipt = json.loads(receipt_bytes)
        expected_receipt = {"version": PROTOCOL_VERSION, "nonce": nonce, "phase": phase,
                            "code_hash": candidate.code_hash, "recipe_hash": candidate.recipe_hash,
                            "output_hash": _hash(output)}
        if receipt != expected_receipt:
            raise ValueError("Candidate output receipt differs from the current invocation")
        frame = read_csv(output_path)
        spec.validate_frame(frame, expected_ids)
        metric = None
        if frozen_actuals is not None:
            if spec.prediction_kind == "probabilities":
                columns = [column for column, _ in spec.probability_columns]
                values = frame.loc[:, columns].to_numpy()
                prediction_classes = tuple(label for _, label in spec.probability_columns)
            else:
                values, prediction_classes = frame[spec.target_column].to_numpy(), None
            value = score_predictions(spec.metric, frozen_actuals, values,
                                      classes=spec.classes, prediction_classes=prediction_classes,
                                      positive_class=spec.positive_class)
            metric = {"name": spec.metric.name, "value": value,
                      "higher_is_better": spec.metric.higher_is_better}
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        raise CandidateExecutionError(f"Candidate predictions failed validation: {error}", execution) from error
    return CandidateResult(output_path, metric, execution, candidate.code_hash,
                           candidate.recipe_hash, phase, receipt_path,
                           _hash(output), len(output), _hash(receipt_bytes))
