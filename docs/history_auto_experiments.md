# Автоматический history: независимый аудит экспериментов

Механизм проверен; режим остаётся экспериментальным opt-in. Синтетика дала **3 выигрыша и 5 побайтных ничьих**, без проигрышей. Ровно три запуска действительно измерили и опубликовали history. Отдельная проверка на публичном TRAIN дала **численную ничью**: неполное предложение было отброшено, history trial не запускался. Этого недостаточно для включения опции по умолчанию.

## Зафиксированный код

- Frozen solver: `a487489145bf8359ba68b82cfba66a42351bc427`; tree: `6e428e38901435ad3082ac8f1d9855070c7efec1`.
- Эквивалентный remote commit [PR #23](https://github.com/NolReaction/alfa-automl-agent/pull/23): `a2d21cd6ccd45cb792dbe145eac8bb62ca1e16a1`. Его tree независимо получен через `git fetch` и совпал.
- Проверены SHA всех 45 solver-файлов, harness, sealed manifest и confirmation freeze. Полные значения, hashes и раздельные стадии: [history_auto_results.json](history_auto_results.json).
- После запусков опубликован test-only head `b8114a36b0b587f3c0915e8a3a23ad07413330b0`. Проверенный локальный diff меняет только `tests/test_history_auto_experiment.py`; runtime и harness экспериментов сохранены.

## Метод и граница данных

- Реальный native v3 `local_only`, одинаковые лимиты: 90 с на arm для синтетики, 300 с для public sanity; `legacy` search, фиксированный weight 0.5. Внешних модельных API requests: **0**.
- В каждом driver request `history_config=null`. Arms различались `history_auto=false/true`; известные ключи из host metadata не передавались как ручная конфигурация.
- Discovery получал точные raw INNER-fit features без ID/y и строк inner-held/outer/test. Сверены CSV SHA, fit-ID SHA, request и physical worker source. Полная top-one schema затем проходила обычный INNER trial.
- Audit labels находились вне solver task directories. Это граница контролируемого data path при одном OS user, не файловая песочница.
- Во время этого аудита не запускались fit, candidate-код или API: проверялись сохранённые артефакты и арифметика predictions. Исходные эксперименты использовали настоящий AutoML, а не fixture baseline.

## Синтетические пары

Δ = RMSE(control) − RMSE(auto); положительное значение означает улучшение. M/I/O/P: реальный measured INNER history trial / inner admission / outer admission / publication history. Выбранный config сам по себе не считается измерением или публикацией.

| Case / seed | Control RMSE | Auto RMSE | Δ | Итог | Discovery | M/I/O/P |
| --- | ---: | ---: | ---: | --- | --- | --- |
| entity_signal_313 | 0.413090377 | 0.379557876 | +0.033532501 | выигрыш | selected | 1/1/1/1 |
| no_history_coverage_313 | 0.185351178 | 0.185351178 | +0.000000000 | побайтная ничья | abstained | 0/0/0/0 |
| colliding_keys_313 | 0.054245930 | 0.054245930 | +0.000000000 | побайтная ничья | abstained | 0/0/0/0 |
| iid_regression_313 | 0.144821567 | 0.144821567 | +0.000000000 | побайтная ничья | skipped | 0/0/0/0 |
| entity_signal_8117 | 0.425124329 | 0.374819833 | +0.050304497 | выигрыш | selected | 1/1/1/1 |
| entity_signal_16417 | 0.487072490 | 0.392027837 | +0.095044653 | выигрыш | selected | 1/1/1/1 |
| colliding_keys_8117 | 0.054741884 | 0.054741884 | +0.000000000 | побайтная ничья | abstained | 0/0/0/0 |
| colliding_keys_16417 | 0.047723602 | 0.047723602 | +0.000000000 | побайтная ничья | abstained | 0/0/0/0 |

Development: seed 313, четыре механизма, **8/8 arms проверены**. Confirmation: seeds 8117/16417, только entity_signal/colliding_keys, **8/8 arms проверены**. Метод и source были запечатаны до confirmation.

Все три `entity_signal` выбрали `entity_key`, выполнили ровно один verified INNER trial, прошли inner/outer admission и опубликовали full-refit history CSV. Снижение audit RMSE составило 8.12%, 11.83% и 19.51% соответственно. `no_history_coverage` отказался из-за отсутствия повторения; `colliding_keys` — из-за отсутствия однозначного прошлого; IID был skipped из-за random validation. Их history trials отсутствуют; пять outputs побайтно совпадают с control.

## Отдельная public TRAIN sanity

Ранее исследованный публичный TRAIN: периоды <5 → период 5. Это не проверка платформенного test/private score.

| Case | Control RMSE | Auto RMSE | Δ | Итог | Discovery | M/I/O/P |
| --- | ---: | ---: | ---: | --- | --- | --- |
| known_public_year5_0 | 47.678458307 | 47.678458307 | +0.000000000 | численная ничья | limited | 0/0/0/0 |

Discovery: **11175 INNER-fit строк × 62 признака**, 1.704 с. Proposer вернул `limited/check_limit`: 24 checked, 137 unchecked, одно partial proposal. Оно правильно отброшено: `selected_config=null`, history attempts = 0.

RMSE равны, но CSV **не побайтно одинаковы**: 5 003 из 18 308 float predictions различаются, max |Δprediction| = 5.68×10⁻¹⁴. IDs совпадают; каждый CSV соответствует собственным full-refit receipt и checkpoint. Это численная ничья. Пользу реально измеренного history на public этот запуск не проверял.

Дополнительно координатор выполнил настоящий CLI из плоского архива: `solution.py task.md --mode offline --time-budget 90 --history-auto`. За 32,914 с получены selected schema, один history trial и корректный итоговый CSV; API requests 0, входные файлы не изменились. Это небольшой panel smoke для протокола упаковки, не дополнительное доказательство качества.

## Проверка доказательств и интерпретация

Заново пересчитаны **18 audit RMSE, 21 outer RMSE и 3 history INNER RMSE**; совпадение с отчётами в tolerance 1e-12. Дополнительная независимая receipt-проверка: **449 saved artifact SHA и все 135 invocation receipts** — development 59, confirmation 62, public 14. Проверены exact rows/IDs, nonce, code/recipe/output hashes, frozen baseline и publication/checkpoint. Integrity gaps не обнаружены.

Во всех восьми синтетических парах baseline recipe и baseline full-refit CSV совпадают побайтно; на public baseline recipe совпал, predictions равны с tolerance 1e-12. В трёх выигравших парах сохранённая разница дополнительно совпала с фиксированной history-смесью того же baseline: 60 применённых строк на пару, без нового fitting.

При этом стартовый поиск завершал и оценивал разное число кандидатов, а в части пар различался порядок local recipes. Это end-to-end включение опции с её стоимостью discovery, **не полностью контролируемый causal experiment с идентичным search execution**. Совпадение конечного baseline локализует механизм трёх наблюдаемых выигрышей, но не устанавливает общий эффект на неизвестных задачах.

## Ограничения и CI

- Малый сконструированный synthetic catalog; confirmation покрывает только две семьи. Ключ — структурная гипотеза, не доказанный ID.
- Outer score используется в admission и не является unbiased estimate. Host-only audit scores показаны отдельно.
- Process inventory недоступен: RSS/CPU time не измерены, фактическое enforcement memory cap не доказано. Локальный cgroup 8 GiB не равен платформе 32 ГБ; platform qualification не установлена.
- Полный CI head `b8114a36` прошёл в обоих push/PR workflows: Linux — 874 core / 227 app tests (skips 2/3), Windows — 874 core / 227 app tests (skips 1/6). Launcher, архив и завершающие шаги успешны; logs и step summaries проверены. Ранние ошибки касались только тестового canonical path и разделителя пути, оба исправлены.
- Raw sealed серии лежат вне git; в репозиторий входят только компактный Markdown и JSON. Архив и проверка восстановления описаны ниже.

## Архив исходных доказательств

Координатор сохранил `history_auto_evidence_20261021.tar.gz`: **15 483 559 байт**, SHA-256:

`9a79423ec26a14248f405694b7da616c294f137a409b87336436f84ac068b59e`

Архив с дедупликацией представляет **6 225 файлов / 2 136 уникальных содержимых**: исходный объём 1 640 548 394 байта, уникальные данные 127 904 043 байта. По сообщению координатора, readback завершён: каждое уникальное содержимое полностью прочитано из архива, его SHA-256 сверен с исходным файлом; каталог и ссылки дедупликации проверены для всех 6 225 логических файлов. Эта проверка упаковки выполнена после независимого аудита метрик и receipts; данные запусков не менялись.

В релиз также входит ускоренный proposer Мурада `534de08` из PR #21. Его совместимость отдельно проверена 106 pure tests и 220 old/new cases. Описанные выше качественные опыты сохраняют исходную привязку к старому frozen solver; им не приписывается повторный запуск на ускоренном source.
