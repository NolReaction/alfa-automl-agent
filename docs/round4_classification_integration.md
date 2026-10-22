# Раунд 4: контракт подключения — integration_pending

База исследования и проверенный host: **`e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`**. Новый runtime пока не подключён к общему scheduler. Настройка по умолчанию остаётся **OFF**; измерения изолированного кандидата не заменяют native off/on и не назначают новый default. Полный native-прогон в этом задании не выполняется — он остаётся отдельной работой интегратора после общего integration SHA.

## Состояние измерений

| Поле handoff | Состояние |
| --- | --- |
| Выбранный variant и полный frozen recipe/hash | Единственный confirmation-финалист `oof`; полный recipe в `data/r4c/confirmation_selection.json`, recipe hash `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc` |
| Screen / confirmation: Macro F1, paired Δ и per-class gates | Confirmation mean: control 0.6151288609 → OOF 0.6302838409, Δ +0.0151549800; все условия пройдены, Worms contribution 0 |
| Фактические wall times, начатые/завершённые физические fits | Confirmation: OOF 203.768–206.808 с, 10 fits против control 21.206–21.591 с, 2 fits; полный screen 21 fits, confirmation 36 fits |
| Full-TRAIN fresh refit, TEST rows, CSV hash | 87 670 fit / 41 166 unscored TEST строк; fresh refit, 10/10 fits, 255.655323 с, правило 1/1/1; CSV SHA-256 `c57583819d69e7cc59eca1859830cc1d3b2d4796160e09127a114cd11521bdbf` |
| Host audit и полный расход серии | 14/14 invocation проверены, inputs unchanged, identity labels совпали; 67 физических fits = 21 screen + 36 confirmation + 10 delivery |
| Решение о возможности экспериментального подключения | Frozen OOF готов к последующей проверке host-интеграции; `integration_pending`, default OFF |

Эти поля основаны на сохранённых результатах плана `docs/round4_classification_plan.md`. Full-refit evidence находится в `data/r4c/trials/oof_full_refit/record.json`, финальная проверка — `data/r4c/audit.json`. TEST-качество не оценивалось, платформенной отправки не было; успешный локальный CSV сам по себе не меняет решение о качестве.

OOF screen действительно обучил/принял/применил одно правило. Из 206.978815 с полного invocation 112.082435 с заняло копирование subsets, 59.949001 с — calibration callbacks, 0.148913 с — guard. Эта измеренная стоимость должна учитываться целиком; ускорение копирования не внесено в замороженную серию. Источник чисел — `data/r4c/trials/oof_seed42/record.json`; окончательное подключение по одному screen не предлагается.

На confirmation 44 OOF затратил 203.768150 с и 10 fits, но не применил правило: `selection_class_drop` из-за B Δ F1 Shellcode −0.0209634255 при пороге −0.02. На 43/137 правила приняты и применены; всего 2 из 3. Поэтому integration cost должен учитывать и дорогой identity fallback. Средняя стоимость OOF 204.874998 с против 21.360580 с у контроля, примерно 9.6×. Guards не менялись; лучший mean относится к фиксированному рецепту на e62, не к полному scheduler.

## Уже реализованный API

```python
from agent.round4_classification_candidate import (
    make_recipe, candidate_source, execute_candidate,
)
from agent.round4_classification_probability import fixed_incumbent_recipe

recipe = make_recipe(
    fixed_incumbent_recipe(),
    classes,                     # точный упорядоченный список из TaskSpec
    variant="oof",              # единственный frozen confirmation-финалист
)
source_text = candidate_source()  # строка source; сохранить в новый .py
# freeze_candidate(source_path, recipe, fresh_directory)
```

Сигнатура factory: `make_recipe(base_recipe, classes, *, variant="identity", asymmetry_pairs=())`. Поддержаны `identity`, `oof`, `frequency`, `asymmetry`, `group_weight`, `extra_trees_asymmetry`. `extra_trees_asymmetry` задаёт собственный фиксированный ExtraTrees 300/depth 18/leaf 4/sqrt/balanced. Для asymmetry variants нужны явные пары; остальные не используют их. Source вызывает `execute_candidate(X_fit, y_fit, X_predict, context)` и возвращает обычный v3 результат `{"predictions": labels}`. Настоящие вероятности остаются внутри adapter/OOF; внешняя метрика сохраняется `Macro F1`, `prediction_kind="labels"`.

Recipe полностью фиксирует base components, class order, feature options, grouped split/OOF seeds, неизменённые `DEFAULT_GUARD_OPTIONS` и caps. `checked_recipe` пересобирает нормализованный recipe и отклоняет изменённые/неизвестные значения. Для действующего измеренного кандидата нельзя незаметно поменять base, guards, пары, число деревьев или caps при подключении. Итоговый handoff содержит полный JSON, а не только variant.

`candidate_source()` — короткий импортирующий source. Его `code_hash` не заменяет хеши импортированных `round4_classification_candidate/probability/features`, `oof_classification` и используемого runtime. Integration SHA и dependency hashes фиксируются вместе с recipe/source. Один и тот же `FrozenCandidate` используется в INNER selection, outer acceptance и full refit; fitted-состояние между фазами не переносится. Runtime не выбирает алгоритм по `phase`, имени задачи, файлу или пути.

## Eligibility и границы данных

- Только classification, Macro F1 labels, точный class order из recipe/context; от 2 до 50 объявленных классов, все присутствуют в текущем fit. Base — один classifier либо равное усреднение двух; probability columns выравниваются по явным именам до усреднения/argmax.
- Непустые fit/predict DataFrame с уникальными столбцами и точным declared feature order. Numeric/categorical роли образуют полное непересекающееся разбиение. ID/target не могут входить в признаки. Максимум **100 000 fit-строк**, **500 000 predict-строк**, **512 выходных признаков**, **10 000 000 ячеек** в каждой матрице после объявленных feature additions.
- `frequency`: используется список категориальных ролей host; существующий transform допускает до **16 столбцов**, до **4 096 категорий на столбец**. Словари `log1p(count_fit)` создаются на текущем fitting subset, unknown=0; prediction rows не меняют counts. В опыте зафиксированы ровно три категориальных признака.
- `asymmetry` / `extra_trees_asymmetry`: до **3** пар с точными ключами `forward`, `reverse`, `quantity`, `unit`; оба входа — разные declared numeric features, повтор пары и коллизия выходного имени запрещены. Семантическую совместимость и единицы подтверждает вызывающий по словарю, runtime её не выводит из имени или корреляции. Публичные имена есть только в экспериментальном recipe. Выход `(forward−reverse)/(forward+reverse)`, оба нуля → 0; отрицательные/nonfinite → NaN, далее штатная missing policy. Масштабирование перед суммой защищает от overflow.
- `group_weight`: размеры групп считаются только по всем текущим raw fit features до augmentation, без labels и prediction rows; `max(.25, n_group**−.5)` нормируется на средний fit-вес. Через `model__sample_weight` веса поступают обеим компонентам, исходные class weights сохраняются.
- Для OOF — raw feature-группы до преобразований, штатный grouped A/B seed 42, три SGKF в A с seed 42; ни одна группа не пересекает границу. `coverage.available` означает возможность структурного split, а не прохождение guard. A/B class support и fit-fold support проверяются OOF preflight **до callbacks**. Требования guard не ослаблены: A/B минимум 100/60 строк и 5 строк каждого класса; admission gain ≥ 0.01, class F1 drop ≤ 0.02.

Нужно различать fallback и контрактную ошибку. Недостаточное безопасное покрытие, незавершённый/поздний OOF или ошибка callback дают identity, если остаётся время на новый полный base fit. Неверная схема/recipe/class order, невалидные probabilities, нарушенные группы/хеши — ошибки, их нельзя выдавать за принятый calibrated результат. При срыве final fit кандидат не публикуется; host сохраняет прежний принятый вариант.

## Реальные deadlines и fits

`execute_candidate` требует конечный исходный `context["deadline_monotonic"]`. Собственный deadline — минимум этого значения и trial cap: **300 с** для обычного варианта, **600 с** для OOF. OOF получает deadline на **120 с раньше** финального. Каждый callback ограничен **90 с** и фактически оставшимся calibration-временем. Final fit использует остаток исходного срока с 5-секундным запасом на возврат/проверки; новый бюджет не создаётся.

`BoundedProbabilityFits` пишет ограниченный request без prediction labels и последовательно вызывает существующий `agent.execution.run_python`; child вызывает `worker_fit`. Поэтому длительный native `.fit()` ограничивается процессным supervisor. Python-проверка времени между этапами — дополнительная, а не единственная защита. Preprocessing, staging, prediction, проверка child metadata/probability hashes также входят во время. Последний child заново обучает модель на всех переданных fit rows.

До OOF действует отдельный **host/experiment admission** по измеренному полному wall time identity adapter `probe`: `4*probe*1.5 + max(120, 2*probe) <= 600`. Это правило не реализовано внутри `make_recipe`/`execute_candidate`; интегратор должен передать его в свой допуск с учётом фактического `state.research_timeout(600)`. Нельзя выдумать probe из чужого рецепта или добавить незапланированный скрытый fit. Если совместимого измерения нет, явно пропустить OOF, сохранив incumbent. Сам runtime независимо соблюдает callback deadlines и final reserve.

Физические fits логируются child-событиями `events.jsonl` перед вызовом `.fit()` (`started`) и после его возврата (`completed`); двухкомпонентный callback — два fit, не один. Если deadline истёк между событием и фактическим `.fit()`, пишется `cancelled_before_fit`: actual started вычитает такую запись, budget charge сохраняет консервативную попытку. `round4_evidence.json` содержит `physical_fits_started/completed`, `physical_fit_count_basis`, все child runs. Timeout не обнуляет уже записанные попытки. Max physical fits: ordinary FixedBlend 2, OOF 10, ExtraTrees 1 на invocation. Итоговый аудит должен сверять события с adapter evidence и явно отличать попытку, обучение и успешное prediction.

CPU максимум 8 потоков, все компоненты/children последовательно. Запрошенный RSS cap 6 GiB не равен измерению. Среда опыта — общий cgroup 8 GiB/8 CPU; недоступный tree RSS/cleanup остаётся unknown. `getrusage(RUSAGE_SELF)` из child metadata — отдельный self-RSS, не peak всего дерева. Конкурсные 32 GB этим не квалифицируются.

## Минимальный запрос интегратору по host e62

Изменения общих файлов здесь **не внесены**. Предлагаемая настройка — `classification_round4_mode: "off" | "on"`, default `off`; такого CLI-флага сейчас нет. Сохранять запрос/variant в `quality_options`; прежняя regression rescue policy не меняется. Допуск: adaptive, открытая `inner_research`, разрешённая task semantics, уже измеренный baseline, доступный INNER, совпадающая grouped policy.

Точка — `agent.research_v3._continue_locally`, после построения `local_candidate_plan`/`seen`, перед обычной warmup continuation. На e62 первые восемь ordinary slots: RF 120, ET 160, ET 200, CB 100/depth 6, CB 100/depth 8, LGB 100/leaves31, LGB 100/leaves63, LogisticRegression 500; затем round-robin refinement и до трёх feature variants. Классификационный `after_warmup` поздний; добавление туда кандидата не гарантирует его достижения.

Одна конкретная предлагаемая замена — имеющийся CatBoost refinement **iterations 400/depth 10/learning_rate .03**. Находить его по семантике в построенной очереди, а не по `local_N` (номер зависит от INNER-ranking/дедупликации). Перенести этот слот первым после baseline в **обеих arms**: `off` исполняет исходный ordinary рецепт, `on` — ровно один frozen measured round4 recipe. Вынутый CatBoost второй раз не запускается; общий deadline и число слотов прежние. Если слот отсутствует/уже измерен или нет допуска/бюджета — явный skip без добавочного trial.

Текущий `run_deadline_continuation` использует один `code_hash`, а `evaluate` замыкает `BASELINE_SOURCE`; просто подставить custom JSON нельзя. Нужен небольшой host-механизм раннего слота с реальными source+recipe+hash, дедупликацией `canonical_recipe_key(recipe, actual_code_hash)` и учётом в том же trial cap. Custom-результат не отправлять в `AdaptiveSearch.observe` как TPE предложение. Сохраняются `_assess_candidate`, host INNER score, `_track_local_candidate`, finalization feasibility, terminal freeze, outer acceptance и штатная публикация.

Для ограничения раннего слота допустимо добавить к `_assess_candidate` необязательный `maximum=1800`, передать в уже существующий `state.measure_inner(..., maximum=maximum)`. Только этот слот задаёт measured cap 300/600, одинаковый для обеих arms и ограниченный исходным reserve. Это предложение интерфейса, не готовый patch. При OOF finalization estimate должен учитывать весь повтор OOF+refit на каждой host-стадии, а не только последний fit.

## Что вернуть после будущего подключения

Интегратор возвращает общий integration SHA, действительную off/on команду и одинаковые условия обеих arms. Полный native paired-прогон не добавляется к текущей изолированной серии. Для будущего опыта нужно подтвердить достигнутый custom source/recipe, host принятие и опубликованную модель; «hook вызван» недостаточно. Запланированный off/on включает одинаковое перемещение контрольного слота, поэтому эффект новой модели не смешивается с разным порядком arms.

Проверяемые артефакты: host receipts nonce/phase/code/recipe/output hashes; `round4_split.json`; `round4_evidence.json` с coverage, rule hash и rules trained/accepted/applied; child metadata, request/probability hashes и fit events; host feedback/refit feasibility; outer/refit receipts и новый CSV hash, совпадающий с final output. Прямое исполнение не создаёт API-клиента; native API0/0 подтверждаются настоящими `run.json` counters. Любые результаты в таблице handoff должны ссылаться на эти записи.
