"""Optional, scratch-trained TabM experiment with nested local validation.

The model is the official Apache-2.0 ``tabm`` package, not a reimplementation.
No optional backend is imported until training or prediction is requested.
This module is deliberately outside the contest/default AutoML portfolio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from agent.metrics import get_metric, score_predictions

TABM_VERSION = "0.0.3"
MAX_INPUT_WIDTH = 4096
MAX_DENSE_CELLS = 100_000_000


@dataclass(frozen=True)
class NeuralConfig:
    epochs: int = 200
    patience: int = 20
    batch_size: int = 512
    k: int = 16
    hidden_size: int = 256
    n_blocks: int = 3
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    dropout: float = 0.1
    max_categories: int = 128
    seed: int = 42
    threads: int = 8

    def __post_init__(self):
        ranges = {"epochs": (1, 1000), "patience": (1, 100), "batch_size": (16, 4096),
                  "k": (1, 32), "hidden_size": (16, 512), "n_blocks": (1, 6),
                  "max_categories": (2, 512), "seed": (0, 2 ** 32 - 1), "threads": (1, 8)}
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, low, high in (("learning_rate", 1e-5, .1), ("weight_decay", 0, 1),
                                ("dropout", 0, .5)):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def availability() -> dict:
    """Cheap package probe; CUDA capability is confirmed when explicitly selected."""
    missing = [name for name in ("torch", "tabm") if importlib.util.find_spec(name) is None]
    versions = {}
    for name in ("torch", "tabm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    reason = "missing_optional_packages" if missing else None
    if not missing and versions.get("tabm") != TABM_VERSION:
        reason = "tabm_version_mismatch"
    return {"available": reason is None, "reason": reason, "missing": missing, "versions": versions}


def _backend(device):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    state = availability()
    if not state["available"]:
        raise RuntimeError("TabM backend unavailable: " + state["reason"] +
                           ". Install PyTorch and requirements.neural.txt in the experiment environment.")
    # Required by deterministic CUDA matmul; set before CUDA initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from tabm import TabM
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use a GPU; CPU fallback is disabled.")
    return torch, TabM


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("TabM experiment time budget exhausted")


def _category(value):
    # Missing and literal text have disjoint representations (including 'NA').
    return "missing:" if pd.isna(value) else "text:" + str(value)


class FoldPreprocessor:
    """Training-only robust scaling and bounded categorical vocabularies.

    Unknown categories map to index 0. Numeric missing flags are always present;
    a validation-only missing value therefore cannot change the fitted schema.
    """
    def __init__(self, numeric, categorical, *, max_categories=128):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        if (not self.numeric and not self.categorical or
                len(set(self.numeric + self.categorical)) != len(self.numeric + self.categorical)):
            raise ValueError("Numeric and categorical columns must form a nonempty, disjoint schema")
        if type(max_categories) is not int or not 2 <= max_categories <= 512:
            raise ValueError("max_categories must be an integer in [2, 512]")
        self.max_categories = max_categories

    def _frame(self, frame):
        if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or
                not set(self.numeric + self.categorical) <= set(frame.columns)):
            raise ValueError("Missing or duplicate feature columns")
        if len(frame) * (2 * len(self.numeric) + len(self.categorical)) > MAX_DENSE_CELLS:
            raise ValueError("Neural dense input budget exceeded")

    def fit(self, frame):
        self._frame(frame)
        if not len(frame):
            raise ValueError("Cannot fit preprocessing on empty rows")
        self.medians_, self.scales_, self.vocabularies_ = [], [], {}
        for name in self.numeric:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            valid = np.clip(values[np.isfinite(values)], -1e30, 1e30)
            median = float(np.median(valid)) if len(valid) else 0.0
            scale = float(np.quantile(valid, .75) - np.quantile(valid, .25)) if len(valid) else 1.0
            self.medians_.append(median)
            self.scales_.append(scale if scale > 1e-12 else 1.0)
        for name in self.categorical:
            counts = frame[name].map(_category).value_counts()
            # Stable frequency ordering; ties never depend on incoming row order.
            ordered = sorted(counts.index, key=lambda key: (-int(counts[key]), key))
            self.vocabularies_[name] = ordered[:self.max_categories - 1]
        self.cardinalities_ = [len(self.vocabularies_[name]) + 1 for name in self.categorical]
        if 2 * len(self.numeric) + sum(self.cardinalities_) > MAX_INPUT_WIDTH:
            raise ValueError("Neural one-hot input width exceeds 4096; reduce category budget/features")
        return self

    def transform(self, frame):
        self._frame(frame)
        if not hasattr(self, "medians_"):
            raise ValueError("Preprocessing has not been fitted")
        numeric = []
        missing = []
        for name, median, scale in zip(self.numeric, self.medians_, self.scales_):
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            mask = ~np.isfinite(values)
            values = values.copy()
            values[mask] = median
            # Clipping before division avoids float overflow for extreme inputs.
            values = np.clip(values, -1e30, 1e30)
            standardized = (values - np.clip(median, -1e30, 1e30)) / scale
            numeric.append(np.arcsinh(np.clip(standardized, -1e30, 1e30)).astype(np.float32))
            missing.append(mask.astype(np.float32))
        numeric_array = np.column_stack(numeric + missing) if numeric else None
        categorical = []
        for name in self.categorical:
            mapping = {key: index + 1 for index, key in enumerate(self.vocabularies_[name])}
            categorical.append(frame[name].map(_category).map(mapping).fillna(0).to_numpy(dtype=np.int64))
        category_array = np.column_stack(categorical) if categorical else None
        return numeric_array, category_array

    def to_dict(self):
        return {"numeric": self.numeric, "categorical": self.categorical,
                "max_categories": self.max_categories, "medians": self.medians_,
                "scales": self.scales_, "vocabularies": self.vocabularies_,
                "transformation": "median_iqr_arcsinh_and_missing_flags_v1"}


def _metric_output(raw, metric, classes):
    if metric.prediction_kind == "labels":
        return np.asarray(classes, dtype=str)[np.asarray(raw).argmax(axis=1)]
    return raw


def _score(raw, y, metric, classes):
    return score_predictions(metric, y, _metric_output(raw, metric, classes),
                             classes=classes, prediction_classes=classes)


class TabMModel:
    """One fitted model; epoch tuning receives only the caller's inner holdout."""
    def __init__(self, *, numeric, categorical, metric="RMSE", classes=(),
                 config=None, device="cpu"):
        self.config = config or NeuralConfig()
        if not isinstance(self.config, NeuralConfig):
            raise ValueError("config must be NeuralConfig")
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.metric = get_metric(metric)
        if self.metric.name not in {"Macro F1", "RMSE"}:
            raise ValueError("The initial TabM experiment supports Macro F1 and RMSE")
        self.classes = tuple(map(str, classes))
        if self.metric.classification and (len(set(self.classes)) < 2 or len(set(self.classes)) != len(self.classes)):
            raise ValueError("Classification needs distinct literal class labels")
        if not self.metric.classification and self.classes:
            raise ValueError("Regression cannot declare classes")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.device = device

    def _tensor_data(self, frame):
        numeric, categorical = self.preprocessor_.transform(frame)
        return (None if numeric is None else self.torch_.from_numpy(numeric),
                None if categorical is None else self.torch_.from_numpy(categorical))

    def _batch(self, tensors, indices):
        return [None if tensor is None else tensor[indices].to(self.device) for tensor in tensors]

    def _predict_tensors(self, tensors, rows, deadline=None):
        torch = self.torch_
        self.model_.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, rows, self.config.batch_size):
                _check_deadline(deadline)
                outputs = self.model_(*self._batch(tensors, slice(start, start + self.config.batch_size)))
                # Average member PROBABILITIES, never logits, for classification.
                prediction = (outputs.softmax(dim=-1).mean(dim=1) if self.metric.classification
                              else outputs[..., 0].mean(dim=1) * self.target_scale_ + self.target_mean_)
                parts.append(prediction.cpu().numpy())
        result = np.concatenate(parts)
        if not np.isfinite(result).all():
            raise ValueError("TabM produced nonfinite predictions")
        _check_deadline(deadline)
        return result

    def fit(self, frame, y, *, tuning_frame=None, tuning_y=None, deadline=None,
            fixed_epochs=None, progress=None):
        if (tuning_frame is None) != (tuning_y is None):
            raise ValueError("Provide both tuning features and labels")
        if fixed_epochs is not None and (type(fixed_epochs) is not int or not 1 <= fixed_epochs <= self.config.epochs):
            raise ValueError("fixed_epochs must be within the configured epoch cap")
        if fixed_epochs is not None and tuning_frame is not None:
            raise ValueError("Fixed epoch refit cannot inspect tuning labels")
        if fixed_epochs is None and tuning_frame is None:
            raise ValueError("Epoch selection requires a separate inner tuning set")
        _check_deadline(deadline)
        torch, TabM = _backend(self.device)
        self.torch_ = torch
        torch.set_num_threads(self.config.threads)
        torch.manual_seed(self.config.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        torch.use_deterministic_algorithms(True)
        self.preprocessor_ = FoldPreprocessor(self.numeric, self.categorical,
                                             max_categories=self.config.max_categories).fit(frame)
        train_tensors = self._tensor_data(frame)
        tuning_tensors = None if tuning_frame is None else self._tensor_data(tuning_frame)
        values = np.asarray(y)
        if values.ndim != 1 or len(values) != len(frame) or not len(values):
            raise ValueError("Training labels must match nonempty training rows")
        if self.metric.classification:
            mapping = {label: index for index, label in enumerate(self.classes)}
            labels = list(map(str, values))
            if set(labels) != set(self.classes):
                raise ValueError("Each training partition must contain every class")
            targets = torch.tensor([mapping[label] for label in labels], dtype=torch.long)
            self.target_mean_, self.target_scale_ = 0.0, 1.0
        else:
            values = values.astype(float)
            if not np.isfinite(values).all():
                raise ValueError("Regression labels must be finite")
            self.target_mean_ = float(values.mean())
            self.target_scale_ = max(float(values.std()), 1e-8)
            targets = torch.tensor((values - self.target_mean_) / self.target_scale_, dtype=torch.float32)
        self.architecture_ = {"n_num_features": 2 * len(self.numeric),
                              "cat_cardinalities": self.preprocessor_.cardinalities_,
                              "d_out": len(self.classes) if self.metric.classification else 1,
                              "k": self.config.k, "n_blocks": self.config.n_blocks,
                              "d_block": self.config.hidden_size, "dropout": self.config.dropout,
                              "arch_type": "tabm"}
        self.model_ = TabM.make(**self.architecture_).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        rng = np.random.default_rng(self.config.seed)
        best_state, best_value, best_epoch, stale = None, None, 0, 0
        self.history_ = []
        stop_reason = "epoch_limit"
        started = time.monotonic()
        for epoch in range(1, (fixed_epochs or self.config.epochs) + 1):
            try:
                _check_deadline(deadline)
                self.model_.train()
                permutation = rng.permutation(len(frame))
                total_loss = 0.0
                for start in range(0, len(frame), self.config.batch_size):
                    _check_deadline(deadline)
                    indices = permutation[start:start + self.config.batch_size]
                    output = self.model_(*self._batch(train_tensors, indices))
                    target = targets[indices].to(self.device)
                    if self.metric.classification:
                        # Each member is supervised independently; mean prediction loss is wrong.
                        loss = torch.nn.functional.cross_entropy(output.flatten(0, 1),
                            target[:, None].expand(-1, self.config.k).reshape(-1))
                    else:
                        loss = ((output[..., 0] - target[:, None]) ** 2).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("TabM training loss is nonfinite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 10.0)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu()) * len(indices)
                row = {"epoch": epoch, "training_loss": total_loss / len(frame),
                       "elapsed_seconds": time.monotonic() - started}
                if tuning_tensors is not None:
                    predictions = self._predict_tensors(tuning_tensors, len(tuning_frame), deadline)
                    value = _score(predictions, tuning_y, self.metric, self.classes)
                    row["inner_metric"] = value
                    improved = best_value is None or (value > best_value if self.metric.higher_is_better else value < best_value)
                    if improved:
                        best_state = {key: tensor.detach().cpu().clone() for key, tensor in self.model_.state_dict().items()}
                        best_value, best_epoch, stale = value, epoch, 0
                    else:
                        stale += 1
                else:
                    best_epoch = epoch
                self.history_.append(row)
                if progress:
                    progress(dict(row))
                if tuning_tensors is not None and stale >= self.config.patience:
                    stop_reason = "early_stopping"
                    break
            except TimeoutError:
                if fixed_epochs is not None or best_state is None:
                    raise
                stop_reason = "time_budget"
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.evidence_ = {"best_epoch": best_epoch, "completed_epochs": len(self.history_),
                          "inner_metric": best_value, "stop_reason": stop_reason,
                          "elapsed_seconds": time.monotonic() - started,
                          "device": str(next(self.model_.parameters()).device),
                          "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
                          "torch_version": torch.__version__, "tabm_version": TABM_VERSION,
                          "deterministic_algorithms": True, "training_rows": len(frame),
                          "tuning_rows": 0 if tuning_frame is None else len(tuning_frame)}
        return self

    def predict_raw(self, frame, *, deadline=None):
        if not hasattr(self, "evidence_"):
            raise ValueError("Model has not been fitted")
        if not len(frame):
            raise ValueError("Cannot predict an empty frame")
        return self._predict_tensors(self._tensor_data(frame), len(frame), deadline)

    def save(self, directory):
        """Export weights and transparent preprocessing metadata, no pickled model."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        weights = directory / "weights.pt"
        self.torch_.save({key: tensor.detach().cpu() for key, tensor in self.model_.state_dict().items()}, weights)
        metadata = {"version": 1, "backend": "official_tabm", "config": asdict(self.config),
                    "metric": self.metric.to_dict(), "classes": list(self.classes),
                    "preprocessor": self.preprocessor_.to_dict(), "architecture": self.architecture_,
                    "target_mean": self.target_mean_, "target_scale": self.target_scale_,
                    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "fit": self.evidence_}
        (directory / "model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
        return metadata


    @classmethod
    def load(cls, directory, *, device="cpu"):
        """Reload this experiment's own checkpoint; do not load untrusted files."""
        directory = Path(directory)
        metadata_path, weights_path = directory / "model.json", directory / "weights.pt"
        if metadata_path.stat().st_size > 2 * 1024 * 1024 or weights_path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("Neural model artifact exceeds its size bound")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") != 1 or metadata.get("backend") != "official_tabm":
            raise ValueError("Unknown neural artifact format")
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != metadata.get("weights_sha256"):
            raise ValueError("Neural weights checksum mismatch")
        schema = metadata["preprocessor"]
        preprocessor = FoldPreprocessor(schema["numeric"], schema["categorical"],
                                       max_categories=schema["max_categories"])
        medians, scales, vocabularies = schema["medians"], schema["scales"], schema["vocabularies"]
        if (len(medians) != len(preprocessor.numeric) or len(scales) != len(medians)
                or not all(type(v) in {int, float} and math.isfinite(v) for v in medians)
                or not all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in scales)
                or set(vocabularies) != set(preprocessor.categorical)):
            raise ValueError("Invalid neural preprocessing statistics")
        for vocabulary in vocabularies.values():
            if (not isinstance(vocabulary, list) or len(vocabulary) >= preprocessor.max_categories
                    or any(not isinstance(v, str) for v in vocabulary)
                    or len(set(vocabulary)) != len(vocabulary)):
                raise ValueError("Invalid neural category vocabulary")
        preprocessor.medians_, preprocessor.scales_ = medians, scales
        preprocessor.vocabularies_ = vocabularies
        preprocessor.cardinalities_ = [len(vocabularies[name]) + 1 for name in preprocessor.categorical]
        if 2 * len(preprocessor.numeric) + sum(preprocessor.cardinalities_) > MAX_INPUT_WIDTH:
            raise ValueError("Neural artifact input width exceeds bound")
        model = cls(numeric=schema["numeric"], categorical=schema["categorical"], metric=metadata["metric"],
                    classes=metadata["classes"], config=NeuralConfig(**metadata["config"]), device=device)
        torch, TabM = _backend(device)
        torch.set_num_threads(model.config.threads)
        expected = {"n_num_features": 2 * len(model.numeric),
                    "cat_cardinalities": preprocessor.cardinalities_,
                    "d_out": len(model.classes) if model.metric.classification else 1,
                    "k": model.config.k, "n_blocks": model.config.n_blocks,
                    "d_block": model.config.hidden_size, "dropout": model.config.dropout, "arch_type": "tabm"}
        if metadata["architecture"] != expected:
            raise ValueError("Neural architecture differs from its recipe")
        if (type(metadata["target_mean"]) not in {int, float} or not math.isfinite(metadata["target_mean"])
                or type(metadata["target_scale"]) not in {int, float}
                or not math.isfinite(metadata["target_scale"]) or metadata["target_scale"] <= 0):
            raise ValueError("Invalid neural target scaling")
        model.torch_, model.preprocessor_ = torch, preprocessor
        model.architecture_ = expected
        model.model_ = TabM.make(**expected).to(device)
        model.model_.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.target_mean_, model.target_scale_ = metadata["target_mean"], metadata["target_scale"]
        model.evidence_ = metadata["fit"]
        return model


def run_neural_experiment(task_path, output_dir, *, device="cpu", time_budget=3600,
                          validation=None, config=None, progress=None):
    """Select epochs on inner rows, refit outer-fit, then score the outer set once.

    Test predictions are from the same outer-fit model. This experiment does not
    publish over the task's submission and does not claim a full-data refit.
    Deadlines are checked between minibatches; no hard OS/VRAM sandbox is claimed.
    """
    from agent.automl import _json_hash, _plain, choose_validation, data_fingerprint
    from agent.search import inner_split_rows
    from agent.splits import split_rows
    from agent.task_spec import load_task, load_task_data, score_holdout

    if type(time_budget) not in {int, float} or not math.isfinite(time_budget) or not 1 <= time_budget <= 43200:
        raise ValueError("time_budget must be finite and in [1, 43200] seconds")
    config = config or NeuralConfig()
    if not isinstance(config, NeuralConfig):
        raise ValueError("config must be NeuralConfig")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    started = time.monotonic()
    deadline = started + time_budget
    output = _plain(Path(output_dir))
    output.mkdir(parents=True, exist_ok=False)
    report = {"version": 1, "status": "running", "candidate": "TabM", "experimental": True,
              "started_at": datetime.now(timezone.utc).isoformat(), "device_requested": device,
              "time_budget_seconds": time_budget, "config": asdict(config),
              "api": {"used": False, "calls": 0}, "pretrained_weights": False,
              "validation_policy": "inner_epoch_selection_then_single_outer_evaluation",
              "resource_limits": "cooperative minibatch deadline; no hard OS or VRAM sandbox",
              "phase": "preparation",
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "package_versions": {}}
    for name in ("torch", "tabm", "rtdl_num_embeddings", "numpy", "pandas", "scikit-learn"):
        try:
            report["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][name] = None

    def save_report():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output / ".report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, output / "report.json")

    save_report()
    try:
        _backend(device)  # Fail explicitly before expensive data work.
        task_path = _plain(Path(task_path))
        spec = load_task(task_path)
        fingerprint = data_fingerprint(spec.data_dir, deadline=deadline)
        task_text = task_path.read_text(encoding="utf-8-sig")
        train, test, _, id_column, target, y, features, numeric, categorical, classification = load_task_data(spec)
        choice, reason = choose_validation(train, test, features, numeric, classification, validation)
        fit_rows, held_rows, split = split_rows(train, y, features, classification, choice)
        inner_fit, inner_held, inner_split = inner_split_rows(train, y, features, classification, choice, fit_rows)
        split["validation_hash"] = _json_hash({"id_column": id_column, "target_column": target,
            "ids": train[id_column].iloc[held_rows].tolist(), "actuals": train[target].iloc[held_rows].tolist()})
        inner_split["validation_hash"] = _json_hash({"id_column": id_column, "target_column": target,
            "ids": train[id_column].iloc[inner_held].tolist(), "actuals": train[target].iloc[inner_held].tolist()})
        report.update(dataset_fingerprint=fingerprint, split=split, inner_split=inner_split,
                      validation_reason=reason, metric_name=spec.metric.name,
                      comparison_key=_json_hash({"dataset_fingerprint": fingerprint, "metric": spec.metric.name,
                                                "higher_is_better": spec.metric.higher_is_better, "split": split}),
                      phase="inner_training")
        save_report()
        options = dict(numeric=numeric, categorical=categorical, metric=spec.metric.name,
                       classes=spec.classes, config=config, device=device)
        inner_model = TabMModel(**options)

        def epoch_progress(event):
            report["latest_epoch"] = event
            save_report()
            if progress:
                progress({"phase": report["phase"], **event})

        # Reserve most time for retraining the frozen epoch count on more rows.
        inner_model.fit(train.iloc[inner_fit], y.iloc[inner_fit], tuning_frame=train.iloc[inner_held],
                        tuning_y=y.iloc[inner_held], deadline=started + .35 * time_budget,
                        progress=epoch_progress)
        report["inner_fit"] = inner_model.evidence_
        report["inner_history"] = inner_model.history_
        selected_epochs = inner_model.evidence_["best_epoch"]
        del inner_model
        report["phase"] = "outer_refit"
        save_report()
        model = TabMModel(**options).fit(train.iloc[fit_rows], y.iloc[fit_rows],
            fixed_epochs=selected_epochs, deadline=started + .90 * time_budget, progress=epoch_progress)
        report["fit"] = model.evidence_
        report["phase"] = "outer_evaluation"
        save_report()
        raw = model.predict_raw(train.iloc[held_rows], deadline=deadline)
        predictions = _metric_output(raw, spec.metric, spec.classes)
        held_ids = train[id_column].iloc[held_rows].tolist()
        actual = train[target].iloc[held_rows].tolist()
        held_path = output / "holdout_predictions.csv"
        spec.holdout_frame(held_ids, actual, predictions, prediction_classes=spec.classes).to_csv(held_path, index=False)
        value = score_holdout(held_path, spec, held_ids, actual)
        np.savez_compressed(output / "holdout_raw.npz", ids=np.asarray(held_ids, dtype=str),
                            actual=np.asarray(actual, dtype=str), predictions=raw,
                            classes=np.asarray(spec.classes, dtype=str))
        test_raw = model.predict_raw(test, deadline=deadline)
        spec.prediction_frame(test[id_column].tolist(), _metric_output(test_raw, spec.metric, spec.classes),
                              prediction_classes=spec.classes).to_csv(output / "test_predictions.csv", index=False)
        _check_deadline(deadline)
        if (data_fingerprint(spec.data_dir, deadline=deadline) != fingerprint or
                task_path.read_text(encoding="utf-8-sig") != task_text):
            raise ValueError("Source task/data changed during training; results are not qualified")
        metadata = model.save(output / "model")
        report.update(status="completed", phase="finished", comparison_eligible=True,
                      metric={**spec.metric.to_dict(), "value": value, "source": "untouched_outer_holdout"},
                      model_artifact={"path": "model", "weights_sha256": metadata["weights_sha256"],
                                      "training_scope": "outer_fit_only"},
                      prediction_scope="same_outer_fit_model; no full_train_refit")
        _check_deadline(deadline)
        save_report()
        return report
    except Exception as error:
        report.update(status="incomplete" if isinstance(error, TimeoutError) else "failed",
                      comparison_eligible=False, error=f"{type(error).__name__}: {error}")
        report.pop("metric", None)
        save_report()
        raise
