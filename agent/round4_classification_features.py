"""Small, schema-declared classification features fitted on current fit only.

No labels enter this module. Frequency counts and duplicate-group weights are
rebuilt for every fold; prediction rows never affect either. Numeric pair
semantics are explicitly attested by the caller in the frozen recipe rather
than inferred from correlations or public column names.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd

from agent.frequency_candidate import fit_frequency_transform


MAX_ASYMMETRY_PAIRS = 3
ASYMMETRY_PREFIX = "__round4_asymmetry_"
GROUP_WEIGHT_FLOOR = .25


class FeatureContractError(ValueError):
    """Feature options or input schema cannot be safely interpreted."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _names(value, name, *, allow_empty=True):
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(item, str) or not item.strip() or "\0" in item for item in value)
            or len(value) != len(set(value)) or not allow_empty and not value):
        raise FeatureContractError(name + " must contain unique explicit column names")
    return list(value)


def normalize_options(options=None):
    """Return detached finite options; no data-dependent choices are made."""
    if options is None:
        options = {"frequency": False, "asymmetry_pairs": [], "group_weight": False}
    if not isinstance(options, dict) or set(options) != {"frequency", "asymmetry_pairs", "group_weight"}:
        raise FeatureContractError("Feature options must declare frequency, asymmetry_pairs and group_weight")
    if type(options["frequency"]) is not bool or type(options["group_weight"]) is not bool:
        raise FeatureContractError("Feature switches must be booleans")
    pairs = options["asymmetry_pairs"]
    if not isinstance(pairs, list) or len(pairs) > MAX_ASYMMETRY_PAIRS:
        raise FeatureContractError("At most three explicit asymmetry pairs are supported")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"forward", "reverse", "quantity", "unit"}:
            raise FeatureContractError("Each pair must declare forward, reverse, quantity and unit")
        for key, value in pair.items():
            if not isinstance(value, str) or not value.strip() or "\0" in value or len(value) > 500:
                raise FeatureContractError("Pair " + key + " must be an explicit nonempty string")
        columns = frozenset((pair["forward"], pair["reverse"]))
        if len(columns) != 2 or columns in seen:
            raise FeatureContractError("Asymmetry pairs must use two distinct columns without repeated pairs")
        seen.add(columns)
    return deepcopy(options)


def _schema(X_fit, X_predict, context):
    if not isinstance(context, dict):
        raise FeatureContractError("An explicit context is required")
    features = _names(context.get("features"), "features", allow_empty=False)
    numeric = _names(context.get("numeric"), "numeric")
    categorical = _names(context.get("categorical"), "categorical")
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
        raise FeatureContractError("Numeric/categorical roles must partition declared raw features")
    technical = [context.get("id_column"), context.get("target_column")]
    task_spec = context.get("task_spec")
    if isinstance(task_spec, dict):
        technical.extend((task_spec.get("id_column"), task_spec.get("target_column")))
    if set(features) & {value for value in technical if isinstance(value, str)}:
        raise FeatureContractError("Technical ID/target columns cannot be input features")
    for frame, name in ((X_fit, "X_fit"), (X_predict, "X_predict")):
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != features:
            raise FeatureContractError(name + " must have exactly the declared feature order")
    if not len(X_fit):
        raise FeatureContractError("Current fit must contain rows")
    return features, numeric, categorical


def asymmetry_feature_name(pair):
    """Stable names depend only on frozen pair declarations."""
    normalized = normalize_options({"frequency": False, "asymmetry_pairs": [pair], "group_weight": False})
    return ASYMMETRY_PREFIX + _hash(normalized["asymmetry_pairs"][0])[:16]


def _asymmetry(frame, pair):
    a = pd.to_numeric(frame[pair["forward"]], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    b = pd.to_numeric(frame[pair["reverse"]], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0)
    zero = valid & (a == 0) & (b == 0)
    positive = valid & ~zero
    result = np.full(len(frame), np.nan, dtype=float)
    result[zero] = 0.
    # Scaling before addition/subtraction avoids overflow for finite extremes.
    scale = np.maximum(a[positive], b[positive])
    scaled_a, scaled_b = a[positive] / scale, b[positive] / scale
    result[positive] = (scaled_a - scaled_b) / (scaled_a + scaled_b)
    return result, {"rows": len(frame), "invalid_rows": int((~valid).sum()), "zero_total_rows": int(zero.sum())}


def _group_weights(frame):
    try:
        # Exact factorization, not hash buckets: the raw features alone define
        # group equality. DataFrame index and any target are not consulted.
        codes, groups = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)
    except (TypeError, ValueError) as error:
        raise FeatureContractError("Raw group features must contain factorable scalar values") from error
    if (codes < 0).any():
        raise FeatureContractError("Raw feature groups could not be assigned")
    counts = np.bincount(codes)
    sizes = counts[codes]
    raw = np.maximum(GROUP_WEIGHT_FLOOR, np.power(sizes.astype(float), -.5))
    weights = raw / raw.mean()
    evidence = {"active": True, "formula": "max(0.25, group_size**-0.5) / mean_fit_weight",
                "group_definition": "all_raw_declared_features_before_augmentation",
                "group_count": len(groups), "duplicate_groups": int((counts > 1).sum()),
                "rows_in_duplicate_groups": int(counts[counts > 1].sum()),
                "largest_group": int(counts.max()), "raw_weight_floor": GROUP_WEIGHT_FLOOR,
                "minimum": float(weights.min()), "maximum": float(weights.max()),
                "mean": float(weights.mean()),
                "group_sizes_sha256": hashlib.sha256(sizes.astype("<i8").tobytes()).hexdigest(),
                "weights_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()}
    return weights, evidence


def augment_features(X_fit, X_predict, context, options=None):
    """Return fit, predict, context, diagnostics, weights without using labels.

    Frequency adds one log1p count for each declared categorical feature (the
    existing transform enforces its 16-column/4096-category limits). Numeric
    interactions add at most three dimensionless directional asymmetries.
    Invalid/negative numeric inputs become NaN for downstream fit-only missing
    value handling; a pair of zeros has neutral asymmetry zero.

    ``options=None`` is the identity representation. No fitted transform or
    vocabulary escapes through diagnostics, and nothing persists across calls.
    """
    options = normalize_options(options)
    features, numeric, categorical = _schema(X_fit, X_predict, context)
    pairs = options["asymmetry_pairs"]
    if any(pair[key] not in numeric for pair in pairs for key in ("forward", "reverse")):
        raise FeatureContractError("Asymmetry columns must be explicitly numeric features")
    additions = [asymmetry_feature_name(pair) for pair in pairs]
    if len(additions) != len(set(additions)) or set(additions) & set(features):
        raise FeatureContractError("Asymmetry feature names collide with the raw schema")
    weights, weight_evidence = (None, {"active": False})
    if options["group_weight"]:
        weights, weight_evidence = _group_weights(X_fit)
    frequency_evidence = {"active": False}
    frequency_additions = []
    if options["frequency"]:
        transform = fit_frequency_transform(X_fit, categorical_columns=categorical, feature_columns=features)
        fit, predict = transform.transform(X_fit), transform.transform(X_predict)
        frequency_additions = list(transform.added_columns)
        frequency_evidence = transform.diagnostics
    else:
        # pandas deep copies arrays, not nested Python object cells. Isolate
        # those too; the existing frequency transform does this itself.
        fit, predict = deepcopy(X_fit), deepcopy(X_predict)
        for original, result in ((X_fit, fit), (X_predict, predict)):
            for column in original.select_dtypes(include="object"):
                result[column] = original[column].map(deepcopy)
    if set(additions) & set(frequency_additions):
        raise FeatureContractError("Feature families produced colliding columns")
    pair_evidence = []
    for pair, output in zip(pairs, additions):
        fit[output], fitting_stats = _asymmetry(X_fit, pair)
        predict[output], prediction_stats = _asymmetry(X_predict, pair)
        pair_evidence.append({**pair, "output_column": output,
                              "fit": fitting_stats, "predict": prediction_stats})
    added = frequency_additions + additions
    new_context = deepcopy(context)
    new_context["features"] = features + added
    new_context["numeric"] = numeric + added
    new_context["categorical"] = list(categorical)
    evidence = {"version": 1, "kind": "round4_fit_only_features", "options": options,
                "options_sha256": _hash(options), "fit_rows": len(X_fit), "predict_rows": len(X_predict),
                "added_columns": added, "frequency": frequency_evidence,
                "asymmetry": {"active": bool(pairs), "formula": "(forward-reverse)/(forward+reverse)",
                              "both_zero_value": 0., "invalid_value": "NaN for downstream fit-only handling",
                              "pairs": pair_evidence}, "group_weight": weight_evidence}
    return fit, predict, new_context, evidence, weights
