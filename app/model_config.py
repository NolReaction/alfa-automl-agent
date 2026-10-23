"""Application boundary for bounded individual models and fixed ensembles."""
from agent.model_config import (
    make_model_config, model_options, normalize_model_params, validate_model_config, model_task_type,
)


def validate_feature_config(value):
    from agent.features import normalize_feature_config
    if not isinstance(value, dict) or set(value) != set(normalize_feature_config()):
        raise ValueError("Нужны все разрешённые настройки признаков.")
    return normalize_feature_config(value)


def feature_evidence(value, *, model=None):
    """Copy optional canonical feature settings without accepting executable fields."""
    result = {}
    if "feature_config" in value:
        result["feature_config"] = validate_feature_config(value["feature_config"])
    if "component_feature_configs" in value:
        components = value["component_feature_configs"]
        expected = (value.get("model_config") or {}).get("parameters", {}).get("components")
        valid_count = (len(components) == 2 if model == "FixedBlend" else
                       2 <= len(components) <= 3 if model == "WeightedBlend" else False) if isinstance(components, list) else False
        if not valid_count or isinstance(expected, list) and len(components) != len(expected):
            raise ValueError("Настройки компонентов должны соответствовать моделям ансамбля.")
        result["component_feature_configs"] = [validate_feature_config(item) for item in components]
    return result


def winner_feature_evidence(value, search):
    result = feature_evidence(value, model=value.get("model"))
    selected = next((row for row in search["candidates"] if row["selected"]), None)
    if selected is not None and value.get("model") is not None:
        if "model_config" in selected and value.get("model_config") != selected["model_config"]:
            raise ValueError("Параметры победителя не совпадают с выбранной попыткой.")
        for key in ("feature_config", "component_feature_configs"):
            if result.get(key) != selected.get(key):
                raise ValueError("Настройки признаков победителя не совпадают с выбранной попыткой.")
    return result
