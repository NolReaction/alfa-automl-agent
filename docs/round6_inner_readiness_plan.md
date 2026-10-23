# Раунд 6: готовность фактического host INNER к замороженному OOF — план до измерения

План зафиксирован до первого замера. После clean commit с кодом и тестами код, пороги и разбиения
не меняются под результат. Если реальный INNER не подходит, это окончательный результат.

## База

- `origin/main` = `114fc2c18595898e0f6ec099e42795eec4af0be9` (merge PR #52 вместе с правками капитана:
  необязательные psutil-счётчики и отказ для вложенных `--classes`, коммит `1c84674`).
- Ветка `work/round6-murad-inner-readiness` создана от этого SHA.
- PR #51 читается только как источник: `origin/work/round5-oof-host-v1` = `25e5967`. Его код не
  импортируется и не запускается, opt-in OOF не включается. Ночной run на `8a1c243` не трогается.

## Откуда берутся таблицы (исходники на базе)

| Таблица | Источник |
| --- | --- |
| Канонический путь host | `agent/research_v3.py:264-265`: `prepare_research` вызывает `ResearchWorkspace(task_path, workspace)` без `frozen_split` |
| TaskSpec и данные | `agent/research_workspace.py:290-297`: `load_task`, `data_fingerprint`, `load_task_data` (все столбцы — строки, `offline_baseline.py:67`) |
| Авто outer | `agent/research_workspace.py:299-301`: `choose_validation` и `split_rows`; для классификации `agent/automl.py:158-163` выбирает `grouped`, если в TRAIN есть дубли признаков. TEST в этой ветке не читается, но `load_task_data` открывает и проверяет `test.csv` |
| Outer fit / holdout | `agent/research_workspace.py:323-338`: `outer_fit_rows`, `outer_holdout_rows`, `research_train = outer_fit_frame` (позиции TRAIN, `reset_index`) |
| INNER fit / held | `agent/research_workspace.py:340-349`: `inner_split_rows(research_train, …)` (`agent/search.py:19-36`) даёт `fit_rows`, `held_rows` — позиции в `research_train`; при отказе `inner_unavailable_reason` |
| Вход PR #51 preflight | `agent/round5_classification_host.py:169-171` (PR #51): `gate.research_train.iloc[gate.fit_rows]`, признаки `research_spec.features`, метки `target_column` |
| Вход identity probe и OOF | там же `:360` и `:397` вызывают `measure_inner` с строками по умолчанию; `agent/research_v3.py:231-238`: `fit_frame = research_train.iloc[fit_rows]` |
| Что видит кандидат | `agent/research_candidate.py:299-300` пишет `fit.csv` (`id`, признаки, target); драйвер `:214` читает его `dtype=str, keep_default_na=False` и передаёт `fit.loc[:, features]` (`:224`) |
| Recipe | PR #51 `:318`: `checked_recipe(make_recipe(fixed_incumbent_recipe(), research_spec.classes, variant="oof"))`; ожидаемый SHA256 на публичной классификации `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc` |
| Аудит fit | `experiment_round5_data_readiness.py:655` `audit_oof_fit` (принят в PR #52): `coverage` (`agent/round4_classification_candidate.py:84-120`), пересчёт границ, gates `fit_oof_guarded_rule` с отказывающимся callback, проверки входов адаптера вероятностей |

Инструмент сверяет свой вход с обоими вариантами: таблицей PR #51 (in-memory) и её копией после
записи и чтения `fit.csv` тем же способом, что `run_candidate`. Сверка: число строк, порядок,
список признаков, построчные hash raw-признаков и меток, SHA256 копии `fit.csv`. Расхождение — отказ.

## Пороги (из кода, не меняются)

| Этап | Условие | Источник |
| --- | --- | --- |
| outer и INNER | каждый класс минимум в 2 raw-группах; все классы в обеих частях | `agent/splits.py:89`, `:96-103` |
| INNER | все классы outer fit в INNER fit и INNER held | `agent/search.py:31-35` |
| A/B | тот же `split_rows` grouped на INNER fit, seed 42 | `agent/round4_classification_candidate.py:101-104` |
| folds | 3 fold `StratifiedGroupKFold`, seed 42, без общих групп | там же `:106-111` |
| gates (по порядку) | A не пуст; класс в A > 0; A ≥ 100; класс в A ≥ 5; класс в каждом fold fit ≥ 1; B не пуст; класс в B > 0; B ≥ 60; класс в B ≥ 5 | `agent/oof_classification.py:296-320`, `agent/classification_decision.py:32-39` |
| размеры child-fit | `max_fit_rows`, `max_classes`, ячейки матрицы | `agent/round4_classification_probability.py` `_inputs` через `audit_oof_fit` |
| PR #51 до preflight | grouped outer и INNER; fit ≤ 100 000, predict ≤ 500 000, признаков ≤ 512, ячеек ≤ 10 000 000 для INNER, outer и полного refit | PR #51 `:80-103` (только для справки, не импортируется) |

Запас = наблюдение − порог для каждого класса и части: allowed fit, A, B, каждый fold fit и held.

## Что делает инструмент

1. Копирует `task.md` и папку данных во временную папку и считает SHA256 каждого исходного файла
   до и после. Исходные файлы не открываются на запись.
2. Строит `ResearchWorkspace` на копии тем же вызовом, что host. Workspace лежит во временной папке.
3. Извлекает фактический INNER fit и сверяет его с входом PR #51 и с копией `fit.csv`.
4. Вызывает неизменённую `audit_oof_fit` на этом fit с context из `research_spec`. Позиции A/B/folds
   берёт из того же вызова `coverage`, который сделал аудит.
5. Проверяет вложенные границы по фактическим позициям TRAIN: outer fit и holdout не пересекаются
   и покрывают TRAIN; `research_train` совпадает с TRAIN по outer-fit позициям; INNER fit и held
   внутри outer fit; аудит получил ровно INNER fit; A/B внутри INNER fit; fold fit/held внутри A.
   Raw-группы считаются по всему TRAIN и не пересекают outer, INNER, A/B и fold-границы.
   Нарушение — отказ (`refused`).
6. Агрегирует support, пороги и запасы; называет guard, который пропускает или блокирует первый fit.
7. Меряет время отдельно и без двойного счёта вложенных таймеров: TaskSpec/data load, outer
   workspace, INNER extraction, coverage, guard/cap audit, export. Вложенные вызовы внутри
   конструктора `ResearchWorkspace` и `audit_oof_fit` измеряются обёртками вызова без изменения
   кода `agent/*`; время этапа — эксклюзивное. Флаг `--control-repeat` делает один контрольный
   повтор модель-free подготовки на новой копии; исходный результат сохраняется.
8. Пишет JSON, CSV и Markdown. Внутри только агрегаты и hashes: нет ID, меток строк, позиций, ключей
   групп, примеров категорий, абсолютных путей и исходных данных. Недоступные счётчики памяти —
   `unknown` со scope процесса.

Решения: `refused` (нарушение контракта или границ), `inner_unavailable` (host не построил INNER),
иначе решение `audit_oof_fit`: `structurally_compatible`, `ineligible` или `unknown`.

## Что не оценивается

Модели не обучаются, identity probe не запускается, API не вызывается. Остаются `unknown` или
`not_evaluated`: Macro F1, gain, class-drop, время identity probe и допуск по стоимости,
время будущего fit и refit, RSS дерева процессов, platform admission, стартовый baseline-поиск
host, LLM-диалог, `dependency_snapshot` и subprocess preflight PR #51. Структурная готовность не
означает, что OOF повысит Macro F1.

## Тесты (без моделей)

Синтетические задачи строятся тем же `ResearchWorkspace`:

- совместимый INNER;
- редкий класс (B < 5) — `ineligible`, `selection_insufficient_class_support`;
- класс пропал из INNER (host replay `frozen_split` уводит класс в outer holdout) — `ineligible`, `tuning_missing_class`;
- класс в одной raw-группе на INNER — `inner_unavailable` или отказ канонического разбиения;
- дубли с разными labels — одна группа, границы не пересекает;
- неверные роли и порядок классов — понятный отказ;
- искусственная утечка границ (подменённые позиции outer/INNER и A/B) — `refused`;
- смена ответов outer holdout и INNER held после заморозки контракта не меняет fit-аудит;
- смена и удаление TEST после построения fit не меняют `audit_oof_fit`;
- исходные файлы побайтово те же; копия в другой папке даёт те же content/config/input hashes;
- в выводе нет ID, категорий, путей и построчных ключей;
- запрещены модельный fit, `run_python`, subprocess кроме `git` и сеть; без psutil экспорт работает.

## Запрещено

Подвыборки и новые seeds для поиска проходящего guard, снижение порогов, удаление редких
классов, изменение folds, выбор другого split, подгонка recipe, правки `agent/*`, `app/*`,
`solution.py`, существующих экспериментов, CLI, зависимостей и чужих заданий.

## Команды измерения

Измерение запускается только из clean commit с кодом и тестами:

    python -m unittest tests.test_round6_inner_readiness tests.test_round5_data_readiness -v
    python experiment_round6_inner_readiness.py --task-md ..\alfa-automl-agent\data\classification\task.md --output-dir ..\round6-reports\classification --control-repeat

Итог: `docs/round6_inner_readiness_report.json`, `docs/round6_inner_readiness_support.csv`,
`docs/round6_inner_readiness_summary.md` и `docs/round6_inner_readiness_results.md` с ответами:
точный fit, будет ли начат первый OOF fit, минимальный запас, стоимость подготовки, что неизвестно.
