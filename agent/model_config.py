"""Bounded offline model settings shared by forms, jobs and worker evidence."""

from __future__ import annotations

from copy import deepcopy
import math


def _integer(name, label, minimum, maximum, *, nullable=False):
    return {"name": name, "label": label, "type": "integer", "min": minimum,
            "max": maximum, "step": 1, "nullable": nullable}


def _number(name, label, minimum, maximum, step):
    return {"name": name, "label": label, "type": "number", "min": minimum,
            "max": maximum, "step": step, "nullable": False}


def _select(name, label, options, *, nullable=False):
    return {"name": name, "label": label, "type": "select", "nullable": nullable,
            "options": [{"value": value, "label": text} for value, text in options]}


def _tree_fields(classification=False, random_forest=False):
    fields = [
        _integer("n_estimators", "Количество деревьев", 50, 600),
        _integer("max_depth", "Глубина дерева", 2, 64, nullable=True),
        _integer("min_samples_leaf", "Минимум строк в листе", 1, 100),
        _select("max_features", "Признаки при разбиении", [
            ("sqrt", "Квадратный корень"), ("log2", "Логарифм по основанию 2"),
            (0.5, "Половина признаков"), (1.0, "Все признаки"),
        ]),
    ]
    if classification:
        weights = [(None, "Без весов"), ("balanced", "Балансировать классы")]
        if random_forest:
            weights.append(("balanced_subsample", "Балансировать в каждом дереве"))
        fields.append(_select("class_weight", "Веса классов", weights, nullable=True))
    return fields


_MODELS = {
    ("baseline", "classification"): {
        "model": "RandomForestClassifier",
        "defaults": {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 2,
                     "max_features": "sqrt", "class_weight": "balanced_subsample"},
        "fields": _tree_fields(classification=True, random_forest=True),
    },
    ("baseline", "regression"): {
        "model": "HistGradientBoostingRegressor",
        "defaults": {"max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                     "learning_rate": 0.1, "l2_regularization": 0.0},
        "fields": [
            _integer("max_iter", "Количество итераций", 50, 600),
            _integer("max_leaf_nodes", "Максимум листьев", 4, 128),
            _integer("min_samples_leaf", "Минимум строк в листе", 1, 100),
            _number("learning_rate", "Скорость обучения", 0.01, 0.3, 0.01),
            _number("l2_regularization", "L2-регуляризация", 0, 10, 0.1),
        ],
    },
    ("extra_trees", "classification"): {
        "model": "ExtraTreesClassifier",
        "defaults": {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 2,
                     "max_features": "sqrt", "class_weight": "balanced"},
        "fields": _tree_fields(classification=True),
    },
    ("extra_trees", "regression"): {
        "model": "ExtraTreesRegressor",
        "defaults": {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 2,
                     "max_features": 1.0},
        "fields": _tree_fields(),
    },
}

# Only model-quality controls enter a saved configuration. Runtime controls,
# file paths, callbacks and executable objectives are never accepted from JSON.
ACTIONS = ("baseline", "extra_trees", "catboost", "lightgbm", "linear")
for _task_type in ("classification", "regression"):
    _classification = _task_type == "classification"
    _cat_defaults = {"iterations": 300, "depth": 6, "learning_rate": 0.05,
                     "l2_leaf_reg": 3.0}
    _cat_fields = [
        _integer("iterations", "Количество итераций", 50, 2000),
        _integer("depth", "Глубина дерева", 3, 10),
        _number("learning_rate", "Скорость обучения", 0.01, 0.3, 0.01),
        _number("l2_leaf_reg", "L2-регуляризация", 0, 100, 0.1),
    ]
    if _classification:
        _cat_defaults["auto_class_weights"] = None
        _cat_fields.append(_select("auto_class_weights", "Веса классов", [
            (None, "Без весов"), ("Balanced", "Балансировать"),
            ("SqrtBalanced", "Мягкая балансировка"),
        ], nullable=True))
    _MODELS[("catboost", _task_type)] = {
        "model": "CatBoostClassifier" if _classification else "CatBoostRegressor",
        "defaults": _cat_defaults, "fields": _cat_fields,
    }
    _light_defaults = {"n_estimators": 300, "num_leaves": 31, "max_depth": -1,
                       "min_child_samples": 20, "learning_rate": 0.05,
                       "reg_lambda": 1.0, "colsample_bytree": 1.0}
    _light_fields = [
        _integer("n_estimators", "Количество деревьев", 50, 2000),
        _integer("num_leaves", "Максимум листьев", 4, 128),
        _select("max_depth", "Глубина дерева", [
            (-1, "Без ограничения"), *[(depth, str(depth)) for depth in (3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)],
        ]),
        _integer("min_child_samples", "Минимум строк в листе", 5, 200),
        _number("learning_rate", "Скорость обучения", 0.01, 0.3, 0.01),
        _number("reg_lambda", "L2-регуляризация", 0, 100, 0.1),
        _number("colsample_bytree", "Доля признаков", 0.5, 1, 0.1),
    ]
    if _classification:
        _light_defaults["class_weight"] = None
        _light_fields.append(_select("class_weight", "Веса классов", [
            (None, "Без весов"), ("balanced", "Балансировать"),
        ], nullable=True))
    _MODELS[("lightgbm", _task_type)] = {
        "model": "LGBMClassifier" if _classification else "LGBMRegressor",
        "defaults": _light_defaults, "fields": _light_fields,
    }
_MODELS[("linear", "classification")] = {
    "model": "LogisticRegression",
    "defaults": {"C": 1.0, "max_iter": 500, "class_weight": None},
    "fields": [
        _number("C", "Обратная сила регуляризации", 0.001, 1000, 0.1),
        _integer("max_iter", "Максимум итераций", 100, 2000),
        _select("class_weight", "Веса классов", [
            (None, "Без весов"), ("balanced", "Балансировать"),
        ], nullable=True),
    ],
}
_MODELS[("linear", "regression")] = {
    "model": "Ridge", "defaults": {"alpha": 1.0},
    "fields": [_number("alpha", "Сила регуляризации", 0.000001, 1000000, 0.1)],
}


def _definition(action, task_type):
    if not isinstance(action, str) or not isinstance(task_type, str):
        raise ValueError("Укажите известный рецепт и тип задачи для настроек модели.")
    definition = _MODELS.get((action, task_type))
    if definition is None:
        raise ValueError("Неизвестный рецепт или тип задачи для настроек модели.")
    return definition


def model_options(task_type: str) -> dict:
    """Expose fresh form metadata; callers cannot mutate the shared allowlist."""
    if not isinstance(task_type, str) or task_type not in {"classification", "regression"}:
        return {}
    return {action: deepcopy(_definition(action, task_type))
            for action in ACTIONS}


def _validate_value(field, value):
    name = field["name"]
    if value is None:
        if field["nullable"]:
            return None
        raise ValueError(f"Параметр {name} не может быть пустым.")
    if isinstance(value, bool):
        raise ValueError(f"Некорректный тип параметра {name}.")
    kind = field["type"]
    if kind == "integer":
        if type(value) is not int:
            raise ValueError(f"Параметр {name} должен быть целым числом.")
        if not field["min"] <= value <= field["max"]:
            raise ValueError(f"Параметр {name} должен быть от {field['min']} до {field['max']}.")
        return value
    if kind == "number":
        if type(value) not in {int, float}:
            raise ValueError(f"Параметр {name} должен быть конечным числом.")
        if not field["min"] <= value <= field["max"]:
            raise ValueError(f"Параметр {name} должен быть от {field['min']} до {field['max']}.")
        return float(value)
    for option in field["options"]:
        expected = option["value"]
        if isinstance(expected, str):
            if type(value) is str and value == expected:
                return expected
        elif expected is not None and type(value) in {int, float}:
            if value == expected:
                return expected
    raise ValueError(f"Недопустимое значение параметра {name}.")


def normalize_model_params(action, task_type, value=None) -> dict:
    """Fill missing controls from the exact original recipe's defaults."""
    definition = _definition(action, task_type)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("Настройки модели должны быть объектом.")
    if set(value) - set(definition["defaults"]):
        raise ValueError("Неизвестные параметры модели. Seed и число потоков задаются приложением.")
    return {field["name"]: _validate_value(field, value.get(field["name"], definition["defaults"][field["name"]]))
            for field in definition["fields"]}


def _quality_options(action, task_type, *, objective=None, target_transform=None):
    """Optional recipe controls never change the legacy parameter dictionary."""
    result = {}
    if objective is not None:
        if (task_type != "regression" or action not in {"baseline", "catboost", "lightgbm"}
                or type(objective) is not str or objective != "mae"):
            raise ValueError("MAE objective is supported only by the regression boosting recipes.")
        result["objective"] = objective
    if target_transform is not None:
        if task_type != "regression" or type(target_transform) is not str or target_transform != "log1p":
            raise ValueError("The only target transform is opt-in log1p for regression.")
        result["target_transform"] = target_transform
    return result


def make_model_config(action, task_type, value=None, *, objective=None, target_transform=None) -> dict:
    """Build a legacy-compatible recipe with explicitly requested quality options.

    Omitted options add no keys to historical configurations. A saved option
    is part of the recipe identity, rather than an unchecked model parameter.
    """
    result = {"model": _definition(action, task_type)["model"],
              "parameters": normalize_model_params(action, task_type, value)}
    result.update(_quality_options(action, task_type, objective=objective,
                                   target_transform=target_transform))
    return result


def validate_model_config(value, action=None) -> dict:
    """Validate saved evidence strictly, without inventing absent parameters."""
    required = {"model", "parameters"}
    optional = {"objective", "target_transform"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - optional
            or any(value[name] is None for name in optional if name in value)):
        raise ValueError("Некорректный отчёт о настройках модели.")
    model = value["model"]
    if not isinstance(model, str):
        raise ValueError("Некорректное название модели.")
    if model in {"FixedBlend", "WeightedBlend"}:
        if set(value) != required:
            raise ValueError("Quality options belong to individual blend components.")
        if action is not None:
            raise ValueError("Ансамбль создаётся только автоподбором, а не отдельным рецептом.")
        parameters = value["parameters"]
        if model == "FixedBlend":
            if (not isinstance(parameters, dict) or set(parameters) != {"weight", "components"}
                    or type(parameters["weight"]) not in {int, float} or parameters["weight"] != 0.5
                    or not isinstance(parameters["components"], list) or len(parameters["components"]) != 2):
                raise ValueError("Нужны два компонента ансамбля с фиксированным весом 0.5.")
        elif (not isinstance(parameters, dict) or set(parameters) != {"weights", "components"}
                or not isinstance(parameters["components"], list) or not 2 <= len(parameters["components"]) <= 3
                or not isinstance(parameters["weights"], list)
                or len(parameters["weights"]) != len(parameters["components"])
                or any(type(weight) not in {int, float} or not math.isfinite(weight) or not 0 < weight < 1
                       for weight in parameters["weights"])
                or not math.isclose(sum(parameters["weights"]), 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Нужны 2–3 компонента ансамбля с положительными весами, сумма которых равна 1.")
        if any(not isinstance(component, dict) or not isinstance(component.get("model"), str)
               or component.get("model") in {"FixedBlend", "WeightedBlend"}
               for component in parameters["components"]):
            raise ValueError("Вложенные ансамбли запрещены.")
        components = [validate_model_config(component) for component in parameters["components"]]
        types = {_task_type_for_model(component["model"]) for component in components}
        if len(types) != 1:
            raise ValueError("Компоненты ансамбля должны решать один тип задачи.")
        weights = ({"weight": 0.5} if model == "FixedBlend"
                   else {"weights": list(parameters["weights"])})
        return {"model": model, "parameters": {**weights, "components": components}}
    choice = next(((recipe, task_type, definition) for (recipe, task_type), definition in _MODELS.items()
                   if definition["model"] == model), None)
    if choice is None:
        raise ValueError("Неизвестная модель в отчёте.")
    recipe, task_type, definition = choice
    if action is not None and action != recipe:
        raise ValueError("Модель не соответствует рецепту запуска.")
    parameters = value["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(definition["defaults"]):
        raise ValueError("В отчёте нужны все разрешённые параметры модели.")
    return make_model_config(recipe, task_type, parameters, objective=value.get("objective"),
                             target_transform=value.get("target_transform"))


def _task_type_for_model(model):
    return next(task_type for (_, task_type), definition in _MODELS.items()
                if definition["model"] == model)


def model_task_type(value) -> str:
    """Resolve a checked individual or frozen blend for report/task validation."""
    config = validate_model_config(value)
    if config["model"] in {"FixedBlend", "WeightedBlend"}:
        return _task_type_for_model(config["parameters"]["components"][0]["model"])
    return _task_type_for_model(config["model"])
