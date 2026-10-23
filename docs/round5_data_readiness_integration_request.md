# Раунд 5: integration request — данные для допуска OOF

Изменения общих файлов в этой работе **не внесены**. Ниже — что инструменту `experiment_round5_data_readiness.py`
пришлось взять через приватные функции или обходом и какой публичный интерфейс это заменит. Каждый пункт
самостоятелен; поведение runtime и пороги guards не меняются.

## 1. `app/profiling.py`: полный сдвиг без лимита 20

Сейчас `_quality` обрезает `numeric_shift` и `new_categories` до 20 записей, `missing_changes` — до 10. На
публичных данных скрыт сдвиг медиан у 39 из 59 числовых признаков регрессии и у 19 из 39 классификации. Инструмент
вызывает приватную `_numeric_shift` для каждого признака и сверяет её с 20 записями профиля.

Запрос: публичная `numeric_shift(name, train, test)` или `profile(task_id, quality_limit=None)` без усечения для
developer-экспорта. UI может по-прежнему показывать первые 20. Без этого инструмент зависит от имени и сигнатуры
приватной функции.

## 2. `app/profiling.py`: TEST-сторона по столбцам

`column()` возвращает пропуски и нечисловые значения только для TRAIN. TEST-доля пустых ячеек известна лишь из
усечённого `missing_changes`, нечисловые и ±inf в TEST не считает никто. Запрос: `missing_count` и
`invalid_count` TEST в тех же определениях, что и TRAIN.

## 3. `agent/round4_classification_probability.py`: проверка входов без fit

Ограничения размера (`max_fit_rows`, `max_predict_rows`, `max_output_features`, `max_matrix_cells`) и
`fit_missing_class` проверяет приватная `_inputs`. Инструмент вызывает её для каждого планового child-fit, чтобы
не копировать логику. Запрос: публичная `check_inputs(X_fit, y_fit, X_predict, context, recipe)` с тем же
`ProbabilitySkipError.reason`.

## 4. `agent/oof_classification.py`: preflight gates без callback

Gates поддержки (`tuning_*`, `fit_missing_class`, `selection_*`) выполняются внутри `fit_oof_guarded_rule` до
первого callback. Чтобы получить их точную причину без обучения, инструмент передаёт X без столбцов и callback,
который сразу отказывается обучать; `callback_error` с этим типом ошибки означает, что gates пройдены. Запрос:
публичная `oof_preflight(y_allowed, classes, groups, split_policy, tuning_rows, selection_rows, folds,
guard_options)`, которая проверяет ту же структуру и возвращает `reason` без staging subsets.

## 5. `agent/round4_classification_candidate.coverage`: причина отказа разбиения

При недоступном grouped split `coverage` сохраняет только `error_type`. Текст причины `split_rows`
(например, «один из классов встречается только в одной группе признаков») инструмент получает повторным вызовом
`split_rows`. Запрос: сохранять в результате `coverage` фиксированный код или текст причины.

## 6. `offline_baseline.load_data`: чтение fit без TEST

`load_data` и `load_task` всегда читают `test.csv`. Fit-only режим не должен зависеть от TEST, поэтому правила
словаря (один id, один target, роли и типы признаков) повторены примерно в 15 строках `read_fit_inputs`. Запрос:
публичная `read_dictionary(path)` или `load_fit_table(...)` без чтения TEST.

## 7. Допуск OOF у host

Перед дорогим OOF-слотом host может вызвать `audit_oof_fit(X_inner, y_inner, context)` на настоящем INNER fit: это
не обучение и не чтение TEST. При решении, отличном от `structurally_compatible`, OOF-слот явно пропускается с
причиной, incumbent сохраняется. На публичной стратифицированной выборке 20% TRAIN кандидат `ineligible`:
в B 4 строки Worms при пороге 5 (`selection_insufficient_class_support`). На outer grouped fit (70 155 строк) — `structurally_compatible`, но в B
только 8 строк Worms.

Стоимость самой проверки нужно учитывать в бюджете. На 87 670 строках весь аудит занял 41.7 с; по профилю больше
90% времени уходит на `StratifiedGroupKFold` внутри `coverage`/`split_rows`. Тот же расчёт выполняет и сам
кандидат внутри `execute_candidate`, то есть внутри своего 600-секундного trial.

## Что остаётся за интегратором

Допуск по времени `4·probe·1.5 + max(120, 2·probe) ≤ 600` требует измеренного identity probe. Прирост Macro F1 на
B, class-drop guard, успешный refit и память проверяются только обучением. Инструмент помечает их `unknown`.
