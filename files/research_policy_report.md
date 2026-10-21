# Политика исследовательского диалога с Qwen

Модуль `agent/research_policy.py` ограничивает контекст исследовательского диалога, сохраняет полезную
обратную связь и выбирает инструкцию для следующего шага Qwen. Он строит **ограниченную проекцию**
истории и не заменяет журнал host, оценку качества или принятие результата. Полная очищенная история
остаётся у host в файлах.

- Базовый коммит: `ccdf49e333495d496ee3a712571199e493d790d4` (`Merge pull request #9 from NolReaction/work/dima-integration`).
- Ветка: `work/nika-research-policy`. Менялись только четыре файла: `agent/research_policy.py`,
  `experiment_research_policy.py`, `tests/test_research_policy.py`, этот отчёт.
- Зависимости не добавлялись; модуль использует только стандартную библиотеку.

## Что решает модуль

В `messages` сохранялись большие результаты инструментов, и каждый шаг добавлял ещё одну сводку журнала.
Длинная первая инспекция повторно оплачивалась последующими запросами. Политика строит запрос из трёх частей:

1. **Стабильный префикс** (system, задача, host-контракт) — после очистки секретов целиком, без сокращений.
   Постоянное пояснение «что считается фактом host» лежит здесь, а не в каждой динамической сводке.
2. **Новейшие группы вызовов** (`assistant` с tool calls и все их результаты — неделимая группа).
   Результаты сжимаются `compact_tool_feedback`; аргументы вызовов (исполненный код) не обрезаются и не
   переписываются, кроме удаления секрета. Старшая из сохранённых групп и непоследние результаты внутри
   группы получают половину бюджета feedback.
3. **Одна динамическая host-сводка** (`HOST STATE v1`): стадия, бюджет, лучшая проверенная метрика, счётчики,
   последние гипотезы и причины отклонения, затем `NEXT STEP` с инструкцией.

Если запрос не помещается в `max_messages_bytes`, лишнее убирается в таком порядке (объёмный вывод уступает
первым, гипотезы и причины отклонения — последними): старые группы, кроме новейшей → feedback новейшей
группы вдвое, затем вчетверо меньше → старые попытки в сводке (до одной) → вся новейшая группа целиком (её
факты остаются в host-сводке вместе с текстом ошибки, `earlier_tool_groups_omitted` это показывает) →
дальнейшее сокращение сводки: старые попытки, детали последней попытки, сама последняя попытка →
последнее plain-сообщение. Если не помещаются сами префикс, контракт и минимальная сводка —
`PolicyBudgetExceeded`; контракт никогда не обрезается.

Чего модуль **не делает**: не запускает код, не читает файлы, окружение и часы, не обращается к сети, не
считает метрику, не угадывает происхождение метрики (оно задаётся `contract["validation"]["scope"]`), не
является песочницей и не заменяет budget guard и защиту от повторного сбоя в runner.

## Интерфейс v1

Имена и сигнатуры совпадают с договорённостью: `PolicyLimits`, `PolicyDecision`, `PolicyContext`,
`PolicyInputError`, `PolicyBudgetExceeded`, `compact_tool_feedback`, `decide_next_action`,
`build_model_messages`. Дополнительно экспортируются `json_size` (тот же подсчёт байтов, что и в лимитах),
константы `REASON_CODES`, `ERROR_KINDS`, `DECISION_STAGES`, `DECISION_ACTIONS` и пороги совета
`LOW_REMAINING_SECONDS = 300`, `LOW_REMAINING_REQUESTS = 1`, `FAILURE_STOP_STREAK = 3`.

- Размеры — UTF-8 байты `json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`.
  Это не токены и не рубли.
- Лимиты — целые числа без `bool`; ноль допустим только для `max_recent_trials` и `max_recent_tool_groups`.
- Неизвестные ключи snapshot/contract игнорируются и не попадают в prompt. Необязательные поля `Trial`
  (`code_hash`, `hypothesis`, `metric`, `reason`, `error_kind`, `error`) при отсутствии считаются `null`;
  `diagnostics` и `resources` могут быть `null` или отсутствовать.
- Сообщения `PolicyInputError` называют путь поля (`snapshot.budget.remaining_seconds`), но не значения.
- `history` принимает только `assistant` и `tool`. Незавершённая группа, неизвестный или повторный
  `tool_call_id`, tool-результат вне группы, нестроковые аргументы — `PolicyInputError`; фиктивные
  результаты не создаются. `tool_call_id` проверяются внутри группы: провайдер вправе повторять их между
  разными ответами.
- Лишние поля сообщений (`reasoning_content` и т. п.) в проекцию не переносятся.

### Диагностика `PolicyContext.diagnostics`

| Поле | Смысл |
| --- | --- |
| `version` | 1 |
| `before_bytes` | размер очищенного несжатого кандидата того же запроса: префикс + вся история + та же host-сводка |
| `after_bytes` | точный размер итоговых `messages` |
| `removed_tool_groups` | сколько групп не вошло в запрос (по окну и по бюджету) |
| `omitted_plain_messages` | сколько assistant-сообщений без tool calls не вошло (последнее остаётся, если группы разрешены) |
| `truncated_fields` | число сокращённых полей: feedback, текст assistant, поля сводки |
| `recent_trials_included` | число попыток в сводке (последняя включается всегда, если лимит > 0) |

Исходных текстов в диагностике нет.

## Коды решений

`PolicyDecision.stage ∈ {inspect, experiment, repair, finalize, finished}`,
`action ∈ {continue, complete, fallback, stop}`, `reason` — стабильный snake_case. Использованные коды:

| Код | Когда | stage / action |
| --- | --- | --- |
| `inspect_with_baseline` | первая инспекция, есть валидный результат | inspect / continue |
| `inspect_without_baseline` | инспекция, результата нет (в том числе после выполненной инспекции) | inspect / continue |
| `first_hypothesis` | нет последней попытки либо ещё нет ни одного проверенного эксперимента | experiment / continue |
| `improved_continue` | host принял улучшение | experiment / continue |
| `not_better_continue` | корректный эксперимент не лучше; лучший результат сохранён | experiment / continue |
| `no_verified_result` | последний вызов не дал проверенного результата, но проверенные уже были | experiment / continue |
| `unverified_continue` | метрика не поддерживается host | experiment / continue |
| `repeated_code` | `code_hash` или нормализованная гипотеза уже были | experiment / continue |
| `repair_execution_error` | `error_kind = execution` | repair / continue |
| `repair_consecutive_failures` | `execution`/`unknown` и ≥ 2 сбоев подряд | repair / continue |
| `repair_resource_limit` | `timeout` или `memory` | repair / continue |
| `repair_evidence_contract` | `evidence`: объясняет недостающий контракт по `required_outputs` | repair / continue |
| `repair_repeated_failure` | `repeated_failure` | repair / continue |
| `repair_unknown_error` | `unknown`: нейтральный совет без догадок о причине | repair / continue |
| `retry_after_api_error` | `api`: состояние эксперимента не потеряно | текущая стадия / continue |
| `finalize_requested` | стадия host `finalize` | finalize / continue |
| `ready_to_finish` | стадия `finalize` и контракт выполнен | finalize / continue |
| `finalize_low_resources` | осталось < 300 с или ≤ 1 запроса | finalize / continue |
| `finish_too_early` | модель просит закончить, контракт не выполнен; в инструкции названо недостающее | стадия по состоянию / continue |
| `contract_fulfilled` | завершение разрешено, метрика проверена host | finished / complete |
| `unverified_completion` | завершение при неподдерживаемой метрике; инструкция называет качество непроверенным | finished / complete |
| `contract_unreachable` | метрика не поддерживается и `allow_unverified = false` | finished / fallback или stop |
| `repeated_failures_keep_result` / `_no_result` | ≥ 3 сбоев подряд | finished / fallback или stop |
| `resources_exhausted_keep_result` / `_no_result` | нет запроса, времени или requests | finished / fallback или stop |

Правила завершения (`complete` возможен только при `finish_requested` и сохранённом валидном результате):
при поддерживаемой метрике `verified_trials >= required_verified_trials`; при неподдерживаемой нужны
`allow_unverified` и хотя бы один unverified trial. Baseline, текст «готово» или напечатанная метрика
недостаточны; корректный `not_better` учитывается. Выполненный контракт признаётся выполненным и при
нулевых ресурсах. Пороги 300 с, 1 request и 3 сбоя — совет policy; дедлайн, budget guard и защита от
повтора остаются у runner.

В инструкциях нет имён `train.csv`, `actual`, столбцов или порядка классов: пути и столбцы берутся из
`allowed_inputs` и `required_outputs`, совместимо с manifest v2 и v3. До стадии `finalize` инструкции не
требуют full-data refit.

## Схема compact feedback

Сохраняются (allowlist): `exit_code`, `timed_out`, `memory_limit_exceeded`, `elapsed_seconds`,
`selection` (`accepted`, `verified`, `metric`, `reason`, `hypothesis`, `diagnostics`), `error`,
`output_check` (`valid`, до трёх ошибок), `previous_output_restored`, числовое подмножество `resources`
(`peak_rss_bytes`, `cpu_seconds`, `memory_limit_bytes`), `stderr`, `stdout`. Пустые потоки опускаются.

Добавляются служебные поля: `feedback_version` (1), `truncated` (поле → исходный размер в байтах; также для
полей, убранных целиком) и `dropped_fields` (число отброшенных неизвестных полей, без их содержимого).
Обрезанный текст содержит маркер `[…cut: …]` с исходным размером.

Приоритет при нехватке байт: обязательный минимум (`feedback_version`, `exit_code`, флаги, `accepted`,
`verified`, `metric`; не помещается — `PolicyBudgetExceeded`) → `error` → `reason` → `hypothesis` →
`elapsed`/`restored`/`output_check` → диагностика (полная; иначе сводка: худшие классы и главные путаницы
вместо матрицы, первые группы срезов; иначе опускается) → `resources` → `stderr` (хвост; после сбоя ему
отдаётся до 45 % остатка) → `stdout` (голова и хвост). Сначала выполняется очистка секретов, затем
сокращение, затем проверка размера — секрет на границе обрезки не оставляет фрагментов.

Текст модели и `stdout` ничего не меняют: `accepted`, `verified`, метрика копируются из `selection`
host (в тесте `stdout` с `score=1.0` не влияет на худшую метрику). Гипотеза остаётся утверждением модели —
это сказано в контракте; непроверенный результат в сводке не превращается в проверенный.

## Воспроизведение

```text
git switch work/nika-research-policy
python -m unittest tests.test_research_policy -v
python -m unittest tests.test_llm tests.test_research tests.test_research_protocol \
    tests.test_research_feedback tests.test_research_validation tests.test_research_policy
python experiment_research_policy.py --label final
python experiment_research_policy.py --label small-limits --max-messages-bytes 24576 --max-feedback-bytes 3072
```

Эксперимент полностью offline: без API, без запуска сгенерированного кода, без данных. Тяжёлые результаты
сохраняются в `data/experiments/nika-research-policy/` только после `git check-ignore`; если путь не
игнорируется, используется временный каталог. Окружение измерений: Python 3.12.3, `openai` 1.68.2,
`pytest` не требуется (тесты на `unittest`).

## Измерения

«Было» — воспроизведение текущего runner: все результаты инструментов остаются в `messages`, перед каждым
запросом добавляется новая сводка `ResearchLedger.context`; тексты system, задачи и инструкций одинаковы с
вариантом «Стало». Сценарии синтетические, но по форме совпадают с результатами runner. Стартовый запрос
«Было» — только system, задача и инструкции; у «Стало» в нём уже есть контракт и host-сводка. Лимиты по
умолчанию (`64 KiB` на запрос, `6 KiB` на feedback, 6 попыток, 2 группы).

| Сценарий | Запросов | Было, всего, Б | Стало, всего, Б | Было, макс. запрос, Б | Стало, макс. запрос, Б | Изменение | Переходы | Факты |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| big_inspection | 4 | 285 247 | 52 566 | 116 020 | 17 581 | −81,6 % | 4/4 | 18/18 |
| many_experiments | 10 | 1 519 420 | 161 659 | 340 652 | 18 182 | −89,4 % | 10/10 | 49/49 |
| long_traceback | 3 | 173 711 | 37 269 | 94 684 | 17 276 | −78,5 % | 3/3 | 15/15 |
| error_then_fix | 3 | 40 771 | 30 052 | 27 784 | 15 250 | −26,3 % | 3/3 | 14/14 |
| multi_call_group | 2 | 49 268 | 26 726 | 45 027 | 20 775 | −45,8 % | 2/2 | 9/9 |
| incomplete_group | 2 | 26 316 | 5 951 | 22 075 | 5 951 | −77,4 % | 2/2 | 4/4 |
| early_finish | 3 | 72 449 | 35 690 | 46 138 | 17 198 | −50,7 % | 3/3 | 14/14 |
| low_time | 3 | 72 473 | 35 486 | 46 153 | 17 168 | −51,0 % | 3/3 | 14/14 |
| unsupported_metric | 2 | 14 573 | 16 977 | 10 332 | 11 139 | +16,5 % | 2/2 | 7/7 |
| holdout_without_actual | 2 | 15 178 | 17 537 | 10 937 | 11 379 | +15,5 % | 2/2 | 13/13 |
| repeated_failures | 4 | 52 652 | 39 290 | 23 518 | 12 173 | −25,4 % | 4/4 | 19/19 |
| **Серия** | 38 | **2 322 058** | **459 203** | 340 652 | 20 775 | **−80,2 %** | 38/38 | все |

(В `incomplete_group` второй запрос не строится: политика отвечает `PolicyInputError`, поэтому «Стало» — это
один запрос, а «Было» относится к обоим. Без этого сценария серия даёт −80,3 %.)

Размер запроса по шагам, `many_experiments` (байты):

| Запрос | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Было | 4 241 | 22 079 | 46 142 | 76 451 | 112 984 | 155 733 | 204 720 | 253 718 | 302 700 | 340 652 |
| Стало | 5 951 | 12 434 | 17 381 | 17 614 | 17 783 | 17 992 | 18 168 | 18 182 | 18 159 | 17 995 |

Рост «Было» накопительный, «Стало» выходит на плато около 18 КБ (диагностика: `removed_tool_groups = 6`
на десятом запросе, `before_bytes = 96 159`, `after_bytes = 17 995`).

Чувствительность к лимитам (`--max-messages-bytes 24576 --max-feedback-bytes 3072`): серия 399 365 Б,
−82,8 %, максимальный запрос 16 103 Б, переходы 38/38, ошибок фактов нет.

Проверки фактов в каждом запросе: размер в пределах лимита; секрет отсутствует; контракт дословно в
префиксе; лучшая host-метрика и последняя гипотеза присутствуют в сводке; причина последнего сбоя
названа; финальная строка длинного traceback сохранена; для manifest без `actual` в сообщениях политики нет
`actual` и `train.csv`. Префикс `[system, task, contract]` идентичен во всех запросах сценария (удобно для
кеша префикса, если провайдер его использует).

## Что нашли и что изменили между итерациями

Каждая итерация измерена тем же воспроизведением (таблицы `results-iter1..4.json`).

| Итерация | Серия «Стало», Б | Изменение | Макс. запрос, Б | Переходы |
| --- | ---: | ---: | ---: | :---: |
| 1. Первая версия | 505 277 | −78,2 % | 25 211 | 37/38 |
| 2. Без дублей и постоянной шапки | 485 568 | −79,1 % | 24 361 | 37/38 |
| 3. Логика после инспекции, бюджет старших групп | 462 789 | −80,1 % | 24 361 | 38/38 |
| 4. Бюджет непоследних результатов группы | 459 203 | −80,2 % | 20 775 | 38/38 |

До первого замера исправлены две ошибки самого эксперимента (счётчик оставшихся requests делал низкий
ресурс в долгом сценарии, а проверка подстроки «actual» срабатывала на слове «factual»); на модуль они не
влияли.

1. **Первая версия.** Замер показал переход после аналитического вызова: политика говорила «исправь
   непроверенный результат» вместо первой гипотезы (1 расхождение из 38).
2. **Дубли и шапка.** Хвост traceback попадал в запрос дважды — в `stderr` tool-результата и в
   `last_trial.error` сводки; диагностика и ресурсы тоже дублировались. Теперь сводка ссылается на уже
   показанное (`error_shown_above`, `diagnostics_shown_above`), если новейший tool-результат содержит это.
   Постоянное объяснение фактов host (~190 Б на запрос) перенесено из динамической сводки в стабильный префикс.
3. **Логика и бюджет.** После выполненной инспекции и пока нет проверенных экспериментов — `first_hypothesis`
   (причина `repair_unverified_trial` заменена на `no_verified_result` и применяется, только когда проверенные
   эксперименты уже были). Старшая сохранённая группа получает половину бюджета feedback; группа, чей минимальный
   feedback не помещается, вытесняется, а не роняет всю сборку.
4. **Группы с несколькими вызовами.** Три крупных результата в одной группе давали 24 КБ; теперь полный
   бюджет получает только последний результат новейшей группы.

## Предположения и ограничения

Измерено: байты запросов в воспроизведении, переходы решений, сохранность перечисленных фактов, поведение
тестов. Это **не** токены, не рубли и не качество ML.

- «Было» — реконструкция поведения runner на синтетических данных, а не повтор реального запуска. Реальный
  контроль Qwen (5 ответов, 6 tool calls, 68 954 reported tokens) здесь не повторялся.
- На коротких диалогах (1–2 группы) политика **больше** прежнего запроса: фиксированная добавка около 1,7 КБ
  (JSON контракта и host-сводка с инструкцией). Выигрыш появляется со второй–третьей группы
  (`many_experiments`: запрос 2 уже на 44 % меньше).
- Реальных запросов к Qwen и к платному API не было: **0 API attempts, usage не расходовался**. Проверка
  совместимости с SDK использует установленный `openai` 1.68.2, loopback-сервер `127.0.0.1`, фиктивный ключ
  и закрывает сервер и сокеты; возвращённый код не выполняется. Она подтверждает, что сервер получает ровно
  `context.messages` (ответ 200) и что безопасное сообщение после HTTP 429 попадает в следующий компактный
  запрос без тела ответа и ключа. Полноценная проверка host-verified ML-экспериментов возможна только после
  интеграции Димой общего guard; потолок первой совместной проверки — 8 requests и мягкий ориентир
  100 000 estimated tokens, командный лимит 600 ₽ в сутки. Правдоподобный ответ Qwen или replay исследованием
  не считается.
- Policy не угадывает, откуда метрика. Для v3 host должен передавать в snapshot только внутренние оценки и
  называть их происхождение в `validation.scope`; диагностика пересылается ровно так, как её дал host.
  Внешний holdout в обратную связь не должен попадать.
- Инструкции не являются песочницей: код Python того же пользователя сам по себе не доказывает отсутствие
  доступа к другим файлам.
- Пороги совета (300 с, 1 request, 3 сбоя), доли бюджета feedback и размеры сводки — выбор по замерам на
  синтетических сценариях; их стоит пересмотреть на реальных журналах.
- Если у провайдера есть счётчик токенов, лимит в байтах стоит откалибровать по `reported tokens`.

## Подключение (три функции)

Host хранит полную очищенную историю как сейчас, строит компактные messages перед каждым запросом и
сохраняет собственные guards.

```python
from agent.research_policy import (
    PolicyBudgetExceeded, PolicyInputError, PolicyLimits, build_model_messages, decide_next_action,
)

LIMITS = PolicyLimits()                                  # значения по умолчанию v1

# ... до запроса: все tool-результаты предыдущей группы уже записаны в history
snapshot = host_snapshot(state)       # нормализованный v1 из ledger/RunState; error_kind назначает host
contract = host_contract(state)       # v1: пути и столбцы из manifest, validation.scope, completion
decision = decide_next_action(snapshot, contract=contract)
if decision.action != "continue":
    return state.finish_with(decision)    # fallback/stop/complete решает runner, у него свои guards

try:
    context = build_model_messages(
        system_prompt=SYSTEM_PROMPT, task_text=state.task_text, contract=contract,
        history=state.dialog,             # только assistant/tool; без system/task/host-сводок
        snapshot=snapshot, limits=LIMITS, redact=state.redact)
except (PolicyInputError, PolicyBudgetExceeded) as error:
    return state.fail_safely(error)       # сохранить принятый результат, не отправлять запрос

state.event("context", **context.diagnostics)
response = client.chat.completions.create(model=get_model(), messages=context.messages, tools=[TOOL], ...)
# state.dialog дополняется как раньше: assistant с tool_calls и tool-результаты (json очищенного result)
```

Что нужно сделать Диме:

1. Преобразовать `ResearchLedger` в snapshot v1 (`error_kind`: `timeout`/`memory` из флагов результата,
   `repeated_failure` из блокировки повторного кода, `evidence` из ошибок evidence, `api` из ошибок
   провайдера, иначе `execution`/`unknown`) и собрать contract из manifest.
2. Убрать добавление `Host research history:` в `messages` и перестать хранить host-сводки в диалоге; в
   `state.dialog` писать только пары assistant/tool.
3. Передать `state.redact` как `redact`; секрет не нужно отдельно передавать в модуль.
4. Вызывать `build_model_messages` только после завершения всей группы tool calls: незавершённая группа даёт
   `PolicyInputError`.
5. Сохранить budget guard, дедлайн, защиту от повторного сбоя и независимую проверку метрики — политика их не
   заменяет.
6. Для v3 класть в snapshot только внутренние оценки и записать их происхождение в `validation.scope`.
7. Реальный прогон Qwen выполнять только после общего guard, в рамках 8 requests.

## Проверки

Команды и фактические результаты (Python 3.12.3, Linux, scikit-learn 1.8.0; `catboost`, `lightgbm`, `optuna`
при первом полном прогоне не были установлены). Полный набор `python dev_check.py` в этой среде не запускался: нужны закреплённые версии из `requirements.txt`.

| Команда | Результат |
| --- | --- |
| `python -m unittest tests.test_llm tests.test_research_feedback tests.test_research_protocol tests.test_research tests.test_research_validation tests.test_task_spec` на базе `ccdf49e` | `Ran 57 tests`, `OK` |
| `python -m unittest tests.test_research_policy` | `Ran 73 tests`, `OK` |
| `python -m unittest tests.test_llm tests.test_research tests.test_research_protocol tests.test_research_feedback tests.test_research_validation tests.test_research_policy` | `Ran 116 tests`, `OK` |
| `python experiment_research_policy.py --label final` | серия −80,2 %, переходы 38/38, ошибок фактов нет, код выхода 0 |
| `python -m unittest discover -s tests` (как в `dev_check.py`), 344 теста на момент запуска | `FAILED (failures=1, errors=5)`: при первом запуске все шесть вызваны отсутствием `catboost`/`lightgbm`/`optuna` (`ModuleNotFoundError`); после их установки `tests.test_build_submission` проходит, а три ошибки `tests.test_models` (`_estimator_type` в адаптере CatBoost) воспроизводятся на чистой базе `ccdf49e`: в среде проверки scikit-learn 1.8.0 вместо закреплённого 1.5.2. К изменениям этой ветки они не относятся |

Тесты проверяют: границы размера и точное `after_bytes`; Unicode и одиночные суррогаты; очистку секрета до
обрезки и секрет на границе; кавычки, обратные слэши, переводы строк; неизменность входов и детерминизм;
группы tool calls (неделимость, неполная группа, неизвестный и повторный id); сохранение host-фактов и
невозможность изменить их текстом модели; несовместимые версии, типы, `NaN`, `bool` вместо числа;
невозможность вместить контракт; правила завершения по контракту; сценарий повторных сбоев со сменой
инструкции (execution → consecutive → fallback), не дублирующий runner-guard (политика только советует;
блокировку повторного кода выполняет runner); отсутствие `actual` и `train.csv` в инструкциях для
manifest v3; чистоту модуля (только стандартная библиотека, без `open`/`exec`/`eval`). SDK-проверка описана
выше.
