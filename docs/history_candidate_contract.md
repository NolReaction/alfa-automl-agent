# Кандидат истории для AutoML: контракт v1

Модуль `agent/history_candidate.py` — самостоятельный адаптер: он берёт уже готовый прогноз регрессионной модели и смешивает его со средним прошлых ответов того же ключа. Модуль ничего не подключает сам. Интеграция, расписание кандидатов и окончательный выбор по метрике задачи — зона Димы. Результаты проверки и рекомендация по весу — в `docs/regression_history_experiments.md` (раздел v2).

## Интерфейс

```python
from agent.history_candidate import fit_history_candidate

candidate = fit_history_candidate(
    task_type="regression",            # "classification" и любое другое значение дают неактивного кандидата
    metric_name="RMSE",                # метрика задачи из agent.metrics: RMSE, MAE, MSE, R2
    feature_columns=tuple(features),   # признаки, которые интегратор передаёт модели; без target и id
    X_raw=train_raw,                   # сырые значения признаков, до prepare_features
    y=y_train,                         # ответы тех же строк (1-D, конечные числа)
    config=config,                     # JSON-конфигурация ниже или None
    blend_weight=weight,               # число в [0, 1]
)
result = candidate.apply(X_raw_query, base_prediction)
result.prediction     # np.ndarray, той же длины и в том же порядке строк
result.applied_mask   # np.ndarray[bool]: где смесь разрешена и применена с ненулевым весом
result.diagnostics    # dict, см. ниже
```

Все аргументы `fit_history_candidate` только именованные и обязательные. `apply` не принимает ответов, ничего не дообучает и не меняет состояние кандидата. Повторный вызов даёт тот же результат. Результат строки не зависит от остальных строк батча.

## Конфигурация

```json
{
  "key_columns": ["State_Factor", "building_class", "facility_type", "floor_area", "year_built", "ELEVATION"],
  "period_column": "Year_Factor",
  "numeric_columns": ["floor_area", "year_built", "ELEVATION"],
  "positive_columns": ["year_built"],
  "ambiguity_policy": "allow"
}
```

- `key_columns` и `period_column` обязательны. Остальные поля необязательны: по умолчанию `[]`, `[]` и `"allow"`.
- Неизвестное поле, дубли в списках, пустые имена, период внутри ключа, `numeric_columns` вне ключа, `positive_columns` вне `numeric_columns`, политика, отличная от `allow` и `past_unique_only`, вызывают `ValueError`. Ошибка возникает, даже если задача классификационная.
- `allow`: смесь для каждой строки с прошлой историей ключа.
- `past_unique_only`: смесь только там, где ключ ни разу не повторялся внутри одного из доступных прошлых периодов.
- Повторы ключа внутри запроса (одного батча) не влияют на решение. Они только видны в диагностике как `rows_repeated_key_in_query`, поэтому прогноз строки не зависит от размера батча.
- `config=None` выключает историю. Знакомые названия колонок историю не включают. Ключ и период берутся только из явной конфигурации. Их нельзя выводить из `id`, имени файла, target или схемы публичной задачи. Совпадение ключа — эвристика, а не доказательство одного объекта.

## Когда кандидат неактивен

`candidate.active` равен `False`, а `candidate.reason` и `diagnostics["reason"]` называют причину:

| Причина | Когда |
| --- | --- |
| `no_config` | `config=None` |
| `classification_task` | `task_type="classification"` |
| `unknown_task_type` | любой другой `task_type`; регрессия по dtype target не угадывается |
| `unsupported_metric` | метрики нет в `agent.metrics` |
| `metric_not_for_regression` | метрика классификационная |
| `history_columns_not_in_features` | ключ или период не входят в `feature_columns` |
| `missing_columns` | ключа или периода нет в `X_raw` |
| `no_usable_fit_rows` | ни у одной обучающей строки нет полного ключа и корректного периода |

У неактивного кандидата `apply` возвращает копию `base_prediction` без изменений. Сохраняются dtype, форма, порядок меток классов и столбцов вероятностей. `applied_mask` — все `False`.

Программные ошибки не прячутся под возврат baseline. Это неверный вес, дубли в `feature_columns`, `X_raw` не DataFrame, рассогласованный по длине или индексу `y`, нечисловой или бесконечный `y` регрессии, прогноз неверной формы. Если `base_prediction` не конечен на строке, где применяется смесь, тоже будет `ValueError`. То же при переполнении суммы ответов.

## Формула и строки без истории

Для строк `applied_mask`:

```text
(1 - blend_weight) * base_prediction + blend_weight * past_mean
```

Прогноз остаётся в исходной шкале цели: без порогов, округления и clipping. Вес — только вес истории, не вес класса и не вес ансамбля. Все остальные строки получают в точности `base_prediction`, без арифметики с `NaN`. При весе 0 весь массив равен baseline. Метрика задачи сохраняется в `diagnostics["metric"]` и не подменяется на RMSE.

`past_mean` — среднее ответов того же ключа из строго более ранних периодов. Ответы того же периода (включая собственный) и будущих периодов не используются. Это так и для строк обучения.

## Сырые значения и нормализация

Ключ и период читаются из `X_raw` до `prepare_features`, потому что после неё числа становятся `float`. Удобнее всего брать признаки из `read_csv(..., dtype=str, keep_default_na=False)`. `Year_Factor` можно убрать из признаков модели, но он должен остаться в `X_raw` и в `feature_columns`.

- **Числовые части ключа и период:** точное `Decimal(str(value).strip())`. `"61242.0"` равно `"61242"`, ничего не округляется, большие целые и близкие десятичные значения различаются. Значение, которое не является конечным числом, считается пропуском. В `positive_columns` пропуском считается и значение ≤ 0.
- **Текстовые части ключа:** `str(value).strip()`, сравнение точное. `None`, `NaN`, пустая строка и строка из пробелов — пропуск. Буквальные `"NA"` и `"?"` — обычные значения. `float` 1.0 в текстовой колонке становится текстом `"1.0"`.
- **Пропуск в ключе:** у строки нет ключа, она ни с чем не совпадает.
- **Пропущенный или нечисловой период:** при `fit` ответ такой строки не запоминается. При `apply` строка остаётся с baseline. Строки не теряются.

## Диагностика

`diagnostics` всегда содержит поля:

- `status`, `reason`, `task_type`, `metric`, `blend_weight`, `ambiguity_policy`, `rows`;
- `fit` со сводкой обучения: число строк, строки с корректным периодом и с полным ключом, запомненные строки, периоды.

У активного кандидата есть ещё:

- счётчики строк: `rows_with_valid_period`, `rows_with_full_key`, `rows_with_past_history`, `rows_with_ambiguous_past_history`, `rows_allowed_by_policy`, `rows_applied`;
- `rows_changed` — сколько прогнозов действительно изменилось;
- `rows_repeated_key_in_query`;
- доли `share_with_full_key`, `share_with_past_history`, `share_with_ambiguous_past_history`, `share_applied`.

## Как вызывать в AutoML

Проверка, одно разбиение `fit → validation` по времени:

```python
model.fit(prepare(train_raw.loc[fit_rows]), y[fit_rows])
base = model.predict(prepare(train_raw.loc[valid_rows]))
candidate = fit_history_candidate(task_type=task_type, metric_name=metric, feature_columns=features,
                                  X_raw=train_raw.loc[fit_rows, features], y=y[fit_rows],
                                  config=config, blend_weight=weight)
score = score_predictions(metric, y[valid_rows], candidate.apply(train_raw.loc[valid_rows, features], base).prediction)
```

Модель и кандидат обучаются на одной и той же части `fit`. Для каждого разбиения кандидат создаётся заново.

Выбор веса, как в `experiment_regression_history.py --mode nested`:

1. Внутри обучающей части строятся последовательные разбиения «периоды `< s` → период `s`».
2. На каждом заново обучаются модель и кандидат: для каждого веса из сетки `(0, 0.25, 0.5, 0.75, 1)`.
3. Выбирается вес с лучшей общей оценкой метрики задачи по всем внутренним строкам.
4. При равенстве (относительная разница до 1e-12) берётся меньший вес.
5. Если в обучающей части меньше двух периодов, вес равен 0.
6. Модель, обученная на всём train, для внутренних прогнозов не используется.

Итоговый прогноз test: модель и кандидат обучаются на всём train, выбранный вес передаётся как `blend_weight`:

```python
model.fit(prepare(train_raw), y)
candidate = fit_history_candidate(task_type=task_type, metric_name=metric, feature_columns=features,
                                  X_raw=train_raw[features], y=y, config=config, blend_weight=chosen_weight)
test_prediction = candidate.apply(test_raw[features], model.predict(prepare(test_raw))).prediction
```

Неактивный кандидат (`candidate.active is False`) вернёт `model.predict(...)` без изменений. Интегратор может сохранить `candidate.reason` в отчёт.

## Что предлагаем Диме для общего кода

Сами общие модули не меняли. Для интеграции нужно:

1. Передавать в кандидат сырую таблицу признаков до `prepare_features`. Сейчас общий конвейер преобразует числа во `float` раньше.
2. Брать `config` из явной настройки задачи, а не из имён колонок. Для публичной регрессии это JSON выше.
3. Держать выбор веса внутри train той же схемы проверки, что и у остальных кандидатов, и сравнивать кандидатов по метрике задачи.
4. Решить, где хранить выбранный вес и причину отказа в отчёте приложения.
