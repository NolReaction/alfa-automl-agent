# Round 4: handoff INNER-контекста Qwen после ревью PR45

Дата: **23 октября 2026, МСК**. База анализа — **`e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`**. Проверен замороженный PR45 HEAD **`29d7564c84b4a9feeacbf96ba3cee9a43b507fde`**; последующие изменения требуют отдельного просмотра.

## Решение сейчас

Рекомендован merge PR45 **как неактивных experimental helpers, mock harness и evidence после CI**. Это рекомендация, а не утверждение о завершённом merge. Production-подключение и новые defaults не одобрены. PR45 не меняет существующие host/CLI/API/model/dependency/build файлы и не добавляет runtime-флагов; этот документ также ничего не активирует.

Измерена только работа протокола на canned mock responses с дешёвым synthetic baseline. Живого API-этапа не было; польза для реального Qwen, равнобюджетного production AutoML и конкурсного балла **не измерена**. Уменьшение числа запросов не считать улучшением качества: stop-after-two потерял полезный поздний кандидат в сохранённом `late_useful` counterexample.

Капитан владеет будущей host-проекцией, точкой подключения и общими host-тестами. ИИ-3 владеет pure LLM/API helpers. Не добавлять неиспользуемый adapter ради наличия кода и не переносить исследовательский цикл в helper.

## Фактический интерфейс PR45

Это отдельная функция; расширения `compact_research_messages(inner_snapshot=...)` в PR45 **нет**. Старый compactor не изменён.

```python
from agent.round4_llm_policy import prepare_round4_messages

prepared = prepare_round4_messages(
    messages,                       # уже очищенный pre-freeze dialogue
    remaining_requests=remaining,   # int 0..128, фактический остаток attempts
    inner_snapshot=snapshot,        # закрытая host-проекция ниже
    experiment_history=history,     # до 12 host observations
)
# PreparedPolicy(messages, policy_version, action, stop_reason, summary)
```

`action`: `create_candidate`, `inspect_one_field` или `stop_local`. Это совет host, не исполненное действие. Исходные SYSTEM, task, tool arguments и tool results сохраняются неизменным prefix; добавление временное, его нельзя накапливать в `messages`. Summary содержит только enums и bounded counters/size observations; это не usage и не публичный sanitizer.

**Текущую stop-policy нельзя включать целиком как default.** В ней остаётся остановка после двух nontrials, хотя её общий default отвергнут отрицательным опытом. Остановка после любого observed exact duplicate также проверена лишь на узком script и не доказывает безопасность отказа от следующих новых гипотез. Evidence projection и политика завершения research принимаются отдельно.

## Закрытая схема snapshot

`agent.round4_llm_evidence.validate_inner_snapshot` принимает version 1, scope `inner_selection` и только перечисленные секции. Producer — trusted host до freeze, не читатель финального `run.json`. Неизвестные optional сведения — `null`; mandatory task contract нельзя угадывать или усекать. Схема проверяет форму, но не удостоверяет истинность заявленного происхождения.

| Секция | Разрешённый фактический источник и проекция |
| --- | --- |
| `version`, `scope` | Константы `1`, `inner_selection`; stage не входит в schema, его проверяет host |
| `task` | `gate.research_spec`: `task_type`, metric только `name/higher_is_better`, `prediction_kind`, literal `classes`, `probability_classes` из probability-column order, `positive_class` |
| `allowed_inputs` | Только семь штатных relative filenames из approved manifest; без путей оригинальных данных |
| `features` | Список `{name, role}` из approved features; role только `numeric/categorical`, ID/target исключены producer |
| `shape` | **Счётчики** `fit_rows`, `validation_rows`, `feature_count`; только размеры INNER fit/held, не массивы позиций и не полный TRAIN/реальный TEST |
| `split` | Только `{strategy, seed}` из `gate.inner_split`; целый split и temporal/group aggregates этот интерфейс не принимает |
| `baseline` | Необязательный reviewed frozen baseline: code/recipe hashes и **полный canonical recipe**, см. ограничения ниже |
| `current_metric` | `state.best_inner`, typed host-measured INNER metric `name/value/higher_is_better`; не model claim и не outer |
| `diagnostics` | Проверенная проекция существующего `inner_feedback`: bounded regression errors либо class support/recall/F1/confusions; без индивидуальных ответов/ошибок |
| `timing` | Только известные scalars: `remaining_seconds`, `final_reserve_seconds`, `maximum_experiment_seconds`, `last_fit_seconds`, `memory_limit_bytes`, `cpu_threads`; отсутствующее остаётся unknown, memory limit не peak RSS |
| `installed_tools` | Реальная host availability check для permitted package names; не предположение по requirements |
| `profile` | Необязательные counts от `profile_inner_fit` на явно переданных **INNER-fit feature-only** rows; не developer-паспорт полного TRAIN |

Snapshot ≤16 KiB; до 64 features/classes, literal ≤200 UTF-8 bytes. Unknown keys и oversize отклоняются, не обрезаются. Optional profile: до 512 rows, 32 columns/16 384 cells, cooperative 0.05s по умолчанию (до 0.25s); slicing и profiling учитываются в исходном deadline, reserve проверяется до этой необязательной работы.

Если `baseline` передан, validator требует `code_hash`, `recipe_hash` и **полный recipe**: canonical sorted compact finite JSON, ≤8 KiB, depth≤8, ≤512 nodes. Hash-only baseline не принимается. Поэтому **не копировать автоматически `state.baseline.recipe`**: сначала review как settings текущего frozen baseline, построенного только по approved fit view, и проверка допустимости bounded serialization. Если происхождение/settings не подходят, передать `baseline=None`; не хэшировать сокращённый summary под видом полного recipe и не менять frozen identity. Наличие baseline recipe в старом dialogue не делает произвольный recipe безопасным новым snapshot.

**Diagnostics должны иметь однозначную принадлежность.** Harness PR45 сочетает `current_metric=state.best_inner` с `diagnostics=report.baseline_inner_diagnostics`. После улучшения это могут быть разные recipes. Будущий host не должен представлять baseline errors как ошибки нового лидера: пока actual schema не содержит reviewed diagnostic identity, при несовпадении источника опустить этот optional block либо отдельно согласовать расширение схемы. Не добавлять самовольные keys в текущий validator.

## История и provenance

History ≤12 rows/12 KiB; closed fields: `scope`, `outcome`, `identity`, `metric`, `elapsed_seconds`, `accepted`, `inspected_field`. `verified_inner` требует actual frozen candidate identity и matching host metric; другие outcomes не могут иметь verified metric или `accepted=True`. Допустимые outcomes определены в `agent.round4_llm_policy.OUTCOMES`; свободные hypothesis/errors/logs не копируются.

В ledger `code_hash` — hash tool script, а **`candidate_code_hash` — frozen source hash**. В новую identity нужен второй и host `recipe_hash`. Recipe settings history optional; hash-only identity здесь допустима. `state.tried_recipes` пополняется до measurement и не доказывает fit. Verified predictions также не доказывают physical training: PR45 правильно показывает untrained zero predictor как verified trial с candidate fits=0.

Не передавать whole `state/gate/report/ledger/manifest/budget`, finalization plan/measurements, source/recipe bytes вне reviewed settings, absolute paths, env, credentials, stdout/error dumps, raw data/predictions/IDs/targets/row positions, outer metrics/answers, `evaluations.final_check` или legacy outer `error_analysis`. `state.clean` заменяет только известный OPENAI_API_KEY и не заменяет provenance/allowlist.

## Точка будущего подключения и отказы

Hook — **внутри `agent.research_v3._request_completion`, в начале `while True`**, до построения свежего `Current host budget` и `V3State.request_allowed`. На каждой реальной попытке, включая retry: проверить `stage == inner_research` и `not dialogue_closed`; построить fresh snapshot/history/remaining; подготовить временный dialogue; добавить штатный budget; проверить **полный фактический payload** существующими size/token/time guards; затем учитывать attempt и отправлять.

Harness `run_arm.prepare` оборачивает `original_request` снаружи, поэтому его snapshot **не обновляется на internal transport retry**. Это ограничение evidence, не готовый production adapter. Не копировать wrapper буквально. Не переносить подготовку в SDK/gateway после guards и не менять model/TOOL/8192 reserve/исходный deadline/SDK retries 0/один глобальный recovery retry.

Disabled/local-only остаются прежними. Invalid scope/schema/pairs или preparation failure не отправляются: безопасный enum и штатная continuation/finalization с исходным остатком времени, без нового retry. Optional missing diagnostics не отменяет valid trial. Source/input/checkpoint integrity failure не маскируется общим catch optional helper. После freeze helper/API запрещены; проигравший outer кандидат штатно не refit, baseline finalization допустима.

Будущий opt-in live gate использует существующий controlled probe: **один общий cap 6 attempts / 30 000 estimated tokens / 600s**, включая stages, retry и оба arms; не отдельные бюджеты на варианты. PR45 mock CLI этого live сравнения не выполняет. Без ключа сохраняется `live_not_measured`; нет оснований объявлять пользу по `OK`, bytes или replay.

## Шесть host mock-gates перед отдельной активацией

1. **Default/offline parity:** без подключения прежние payload/counters/reasons; local-only не вызывает helper/client; модель и rollout не меняются.
2. **Закрытая INNER-проекция:** sentinel outer/TEST/IDs/secrets/private paths не попадают в snapshot/summary; обязательная class/probability схема полная, unknown не превращается в ноль.
3. **Фактический payload:** snapshot/hint и fresh budget проходят исходные guards вместе; oversize блокируется до SDK; SYSTEM/tool-пары и исходный dialogue неизменны.
4. **Fresh retry и cap:** при mocked transport failure второй snapshot отражает новые remaining/time; общий cap или deadline запрещает следующий send; malformed paid response не запускает transport retry.
5. **Identity/diagnostics/terminal:** candidate hash не путается с tool hash; baseline diagnostics не выдаются за leader errors; после freeze нет helper/API; INNER winner, проигравший outer, корректно публикует baseline без forced loser refit.
6. **Отказы:** bad pair/scope/NaN/helper exception дают zero sends и штатную финализацию; optional absence допустимо, integrity failure не скрывается. Никакой перезаписи ранее верифицированного результата emergency prior из-за ошибки контекста.

Эти mock-gates доказывают интеграционные ограничения. Польза живых предложений и fresh candidate publication требуют отдельного bounded native/live evidence; merge неактивных модулей сам по себе этого не доказывает.
