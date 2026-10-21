# Явный кандидат истории в Research v3

История остаётся экспериментальной опцией. Без настройки прежний offline AutoML
и диалоговый путь сохраняют свои модели и порядок поиска. Schema-proposer Мурада
и OOF-калибровка другого исполнителя в эту интеграцию не входят.

## Что делает агент

После подготовки и измерения исходного baseline на INNER host может проверить
ровно один дополнительный рецепт. Он содержит неизменённый `base_recipe`, явный
history config и фиксированный `blend_weight`. Код базовой модели дополняется
обёрткой `agent.history_runtime.history_candidate_source`. Базовая модель получает
копии входов и исходный recipe; адаптер получает сырые строки X_fit и только y_fit.

Адаптер смешивает базовый прогноз с прошлым средним этого ключа. Используются
только периоды строго раньше прогнозируемой строки. Ответы query, того же
периода и будущего не подставляются. Для неизвестного ключа остаётся исходный
прогноз; строки и ID не переставляются. Вес не подбирается повторно внутри fit.

Допуск v1 требует:

- объявленную регрессию и все нужные колонки среди признаков;
- временную политику и в outer, и в INNER, с одним и тем же периодом из config;
- измеренный исходный baseline, доступный INNER и ненулевой вес;
- остаток исходного бюджета сверх резерва финализации.

Неподходящая задача получает явный skipped reason, без дополнительного history-fit.
Нет автоматического вывода ключа по названию колонок. При нулевом покрытии история
может быть измерена и отклонена как `not_better`: её фактическое обучение учитывается
во времени. Конфиг сам не меняет уже выбранную валидацию.

Отбор выполняет штатный `_assess_candidate` по INNER. Зафиксированные code/recipe
hashes используются затем для outer и полного refit. Ошибка кандидата не меняет
frozen baseline. Outer-оценка появляется только после закрытия диалога и не
возвращается следующей гипотезе. Финальный победитель выбирается по outer,
поэтому его оценка не является независимой несмещённой оценкой системы.

## Запуск

Python API:

```python
from agent.research_v3 import run_agent_v3

config = {"key_columns": ["entity"], "period_column": "period"}
run_dir = run_agent_v3(
    task_path, None, time_budget=90, local_only=True,
    history_config=config, history_weight=0.5,
)
```

`local_only=True` исключает создание клиента и запросы к модели. Подготовка,
измерение истории, обычное локальное продолжение, outer и refit расходуют один
deadline. Успешная проверенная публикация отмечается как completed/local_only;
это завершение локального решателя, а не работа Qwen.

CLI принимает inline JSON либо `@путь-к-JSON`. Для PowerShell удобнее файл,
чтобы кавычки JSON не зависели от версии оболочки:

```powershell
@{key_columns=@("entity"); period_column="period"} |
    ConvertTo-Json | Set-Content -Encoding UTF8 history.json
.\.venv\Scripts\python.exe solution.py task.md --mode offline --time-budget 90 --history-config '@history.json' --history-weight 0.5
```

`entity` и `period` здесь только пример, не публичная схема и не встроенные имена.
Официальный ZIP не переносит JSON-конфиги: файл нужен для явных локальных опытов.
В диалоговом режиме те же опции передаются вместе с обычным клиентом.
`--validation`/`--time-column` с этой опцией отклоняются: v3 фиксирует свою auto
политику и не должен молча игнорировать пользовательское разбиение.

## Диагностика и проверка

`run.json.history_research` хранит requested/active, статус и причину, config/вес,
baseline/candidate hashes и host feedback с областью `inner_selection`.
`history_successful_experiments` считает проверенные history-пробы отдельно от
LLM и обычных локальных проб. В API передаётся только INNER-сводка истории.
Окончательные публикация, receipts и ошибки refit используют обычные поля v3.

Команды профильных проверок:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_history_runtime tests.test_history_cli tests.test_history_research -v
```

Протокол и результаты парных измерений публикуются в
[отдельной ветке доказательств](https://github.com/NolReaction/alfa-automl-agent/blob/work/history-runtime-evidence-v1/docs/history_runtime_experiments.md).
Настоящий AutoML baseline, общий wall-time budget, все отказы и неизменность
исходных данных проверяются стендом; fixture baseline в unit-тесте не является
подтверждением улучшения качества. До результата квалификации default не меняется.
