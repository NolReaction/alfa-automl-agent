"""Bounded offline model settings shared by forms, jobs and worker evidence."""

from __future__ import annotations

from copy import deepcopy


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


def _definition(action, task_type):
    if not isinstance(action, str) or not isinstance(task_type, str):
        raise ValueError("Укажите известный рецепт и тип задачи для настроек модели.")
    definition = _MODELS.get((action, task_type))
    if definition is None:
        raise ValueError("Настройки моделей доступны для baseline и ExtraTrees в классификации или регрессии.")
    return definition


def model_options(task_type: str) -> dict:
    """Expose fresh form metadata; callers cannot mutate the shared allowlist."""
    if not isinstance(task_type, str) or task_type not in {"classification", "regression"}:
        return {}
    return {action: deepcopy(_definition(action, task_type))
            for action in ("baseline", "extra_trees")}


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


def make_model_config(action, task_type, value=None) -> dict:
    return {"model": _definition(action, task_type)["model"],
            "parameters": normalize_model_params(action, task_type, value)}


def validate_model_config(value, action=None) -> dict:
    """Validate saved evidence strictly, without inventing absent parameters."""
    if not isinstance(value, dict) or set(value) != {"model", "parameters"}:
        raise ValueError("Некорректный отчёт о настройках модели.")
    model = value["model"]
    if not isinstance(model, str):
        raise ValueError("Некорректное название модели.")
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
    return make_model_config(recipe, task_type, parameters)
