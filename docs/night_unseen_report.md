# Незнакомые табличные задачи: квалификация контрактов

Исходная база: `390829239bfb245e70d866c02d4cd69d51d4c0d9`.
Ветка: `work/night-unseen-v1`. PR готов к review; перед merge остаётся один
блокер совместимости legacy V2 (см. ниже). Работа ограничена данными, форматом задачи,
метриками, валидацией и независимым developer-only стендом.
Поиск моделей, общий CLI, recovery, приложение, зависимости, упаковщик и
отправленный конкурсный ZIP не изменяются.

## Источники и воспроизводимость

Изучены README, официальный список библиотек, действующий CLI, TaskSpec,
парсер метрик, преобразования, метрики, два существующих benchmark-модуля,
документация платформы и восстановления. Baseline организаторов доступен
во вложении под именем `c5usiwveoqktnd668wdt5mcis50ovg8q.txt`:
SHA-256 `ecf66e8d7c054624eeca710c13e2b82c1ac22a03314db8a45ff1778fa16e372d`.
Он не добавлен в репозиторий и не запускается с реальным API.

Рабочий Python 3.12.14; установлены существующие project pins, в том числе
NumPy 1.26.4, SciPy 1.12.0, pandas 2.2.3, sklearn 1.5.2,
CatBoost 1.2.8, LightGBM 4.6.0, psutil 6.1.1. Это проверка используемых
зависимостей, а не воспроизведение всего Docker-образа организаторов.
Полный список версий фиксируется в manifest каждого запуска.

## Предварительно заданная матрица

Каталог версии 1, seed `81221`, закреплён до внешних solver-оценок:

| Группа | Количество | Интерпретация |
| --- | ---: | --- |
| Валидные | 19 | Поддержанные табличные данные и однозначная схема |
| Трудные | 3 | Ограничения split, дисбаланса и стоимости |
| Некорректные | 10 | Ожидается понятный отказ, а не ложный успех |
| Метаморфные | 3 | Проверка инвариантов, без заявления о новых семействах |
| Scale/budget probes | 3 | До 1600 train / 320 test, без 32 GiB стресс-теста |

Новые механизмы включают перепад давления через сопротивление/клапан,
составной счётный спрос и поставки, пороговую усталость, насыщаемое связывание,
логические маршруты, кольцевые границы, каскадную классификацию и риск отказа.
Они отличаются от механизмов прежних шести benchmark-семейств и трёх MAE.
Варианты формата внутри семейства не объявляются независимыми задачами.
Два metamorphic-случая явно происходят от существующего benchmark, третий —
от нового routing-семейства с буквальной Markdown-пунктуацией классов.

13 положительных случаев выделены для настоящего обучения: все девять
метрик, бинарные и многоклассовые вероятности, обратный positive class `NA`,
пустые numeric/category. Отдельно предусмотрены три точных default CLI
запуска: `flow_rmse`, `logic_accuracy`, `checker_multiclass_logloss`.

Контрактные проверки остальных комбинаций не приравниваются к обучению.
Неподдержанные ranking, NLP, CV и произвольные CSV dialects не добавляются
молча. Отсутствие их поддержки не зарегистрировано как дефект.

## Независимый runner

`qualify_unseen_tasks.py` получает явный `--solver-root` и запускает выбранный
checkout только в дочернем процессе. Процесс аудитора не импортирует код
этого checkout. По умолчанию выполняется только preflight; запуск и re-audit
явные. Каждый план получает новый каталог, старый submission не допускается.

Генератор пишет в задачу только task.md и четыре CSV. Внешние ответы,
ожидаемый контракт и исходные хеши хранятся отдельно у аудитора. Путь к
ответам отсутствует в solver argv. Среда очищается по allowlist; конкурсные
API-настройки не передаются. Обычные процессы под тем же пользователем
сохраняют доступ к файловой системе: это граница потока данных, не OS sandbox.

План закрепляет исходный SHA, dirty flag, хеши исходников и входов, версии,
случаи, метрики, команды и конечные бюджеты. Проверка итогового CSV независимо
сверяет заголовок, literal IDs, порядок test, классы/вероятности и пересчитывает
метрику по внешним ответам. Status полноценной модели требует соответствующего
checkpoint/receipt и реальных артефактов полного refit. Target prior считается
отдельным допустимым fallback с неподтверждённым качеством.

Внешний timeout стенда — цензурированное наблюдение. Например, остановка
default solver через 90 s не доказывает нарушение платформенных трёх часов.
RSS/CPU обозначаются недоступными, если процессная среда не даёт их измерить.
Все обучения последовательны, до восьми потоков; для первой серии установлен
общий конечный лимит не более 3600 s с отдельными ограниченными перепроверками.

## Подтверждённые дефекты и минимальные воспроизведения

База проверяется теми же новыми regression-тестами из candidate checkout,
но импорты идут из отдельного неизменённого baseline worktree. Число failing
subtests не считается числом самостоятельных дефектов.

| ID | Воспроизведение в базе | Причина / исправление | Regression test | Повторная проверка |
| --- | --- | --- | --- | --- |
| U01 | Лишнее CSV-поле становится индексом и сдвигает ID; короткая/пустая запись дополняется/пропускается | Перед pandas добавлена строгая проверка количества полей и CSV syntax | `test_night_data.StrictRecordTests` | OK, targeted tests |
| U02 | Complex/date/timedelta/bool принимаются как числовые прогнозы; мнимая часть может исчезнуть | Общая проверка вещественных представлений до cast, используется также TaskSpec | `test_night_data` и `test_night_task_contract` | OK, targeted tests |
| U03 | Настоящие `pd.NA`/`NaT` становятся строковыми классами | Реальные пропуски отклоняются; буквальные строки `NA`, `NaT`, `<NA>` сохраняются | `test_night_data` | OK, targeted tests |
| U04 | Непустая пробельная категория/текст считается missing | Только пустая строка означает строковый пропуск | `test_night_data` | OK, targeted tests |
| U05 | Полностью пустая категория downcast в float ломает строковый imputer | Поддерживается object dtype перед заполнением | `test_night_data` с настоящим pipeline fit | OK, targeted tests |
| U06 | Буквальное `__MISSING__` сливается с generated imputer token | Инъективное экранирование токена и своего escape-пространства | `test_night_data` | OK, targeted tests |
| U07 | `Metric: RMSE and mean absolute error` принимается как RMSE | Проверяются также полные алиасы вторых метрик; повтор того же алиаса в скобках допустим | `test_night_task_contract` | OK, targeted tests |
| U08 | В class/positive declarations теряются `__`, `**`, backticks | Markdown-нормализация не меняет буквальные JSON/CSV payloads | `test_night_task_contract` | OK, targeted tests |
| U09 | Generic target-named probability column ошибочно конфликтует с positive class | Сопоставление именного class-column отделено от generic target-column | `test_night_task_contract` | OK, targeted tests |
| U10 | Multiclass AUC averaging берётся из постороннего примера; supplementary metric вызывает ложный отказ | Averaging читается только из ведущего предложения основной декларации | `test_night_task_contract` | OK, targeted tests |
| U11 | Legacy validator принимает пустой test/submission и противоречивые dictionary/template/task | Для известных метрик используется TaskSpec, структурные проверки усилены | `test_night_validation` | OK, targeted tests |
| U12 | Изменённый test без ID даёт `KeyError`, а пустой test может приниматься | Проверяются schema и непустые уникальные test IDs; возвращается invalid report | `test_night_validation` | OK, targeted tests |

Публичные сигнатуры, TaskSpec keys и сериализация сохранены. Конвенция Macro F1
по отсутствующим классам не менялась: недостаточно основания объявлять её
ошибкой данного контракта. Отсутствующая/неподдержанная метрика сохраняет
legacy structural validator; такой путь не подтверждает качество ML.

## Замороженные версии и фактическая серия

Сравниваются база `390829239bfb245e70d866c02d4cd69d51d4c0d9` и runtime candidate
`461d8c97261f7bf41ea016775e233dc32a29ab84` (tree
`ecf1d2c17b0bb3fdc458e512a57b45f8083a2230`, совпавший с локальным checkpoint
`f2eb10fa046b7e50a71d572d684001ccb814d0ee`). Core/default и три сохранённых
результата первой extra-серии выполнены на этом точном публичном SHA.
Следующий публичный `78db10f489d5f759fa67d7aafd2a2947dd107278` исправляет только
переносимость тестов UTF-8/Windows. Публичный
`3281a863f1f45b92b78309db9eee36bc7a457fa0` (tree
`c21eec61aa5730382a2f41357ce1b31f85ac162d`) меняет только независимый runner и
его тесты после `78db`; на нём продолжена вторая серия. Побайтно проверено,
что шесть изменённых в PR runtime-модулей, `solution.py` и `unseen_cases.py`
совпадают с `461d8c9`. Runtime модели и генераторы во время серии не менялись.
PR: https://github.com/NolReaction/alfa-automl-agent/pull/15 .

Все основные планы используют seed `81221`: это публичный рабочий набор
квалификации и разработки. Закрытая подтверждающая серия пока не выполнялась;
эти результаты не объявляются её прохождением. Перед запуском проверены чистый
checkout, SHA и хеши всех исходников. Независимый review сверил все 38 пар:
case IDs, определения, task.md, четыре CSV и sealed audit JSON побайтно
совпадают. Фактические схемы, positive class, порядок колонок/классов и
направление метрик также сверены независимо. Снимок проверки:
`contract_independent_check_81221.json` под каталогом результатов.

Дочерние процессы запускаются последовательно, среда задаёт максимум восемь
вычислительных потоков. В первой серии общий предел 3000 s; core использует
30 s solver / 75 s external, default — 90 s external, contract — 30 s external.
После сбоя инфраструктуры отдельно зафиксирована конечная вторая серия с
пределом 600 s: девять незавершённых extra-cases, четыре scale-probes и два
плана намеренного 8-секундного прерывания, пропущенные до запуска solver.
Перед ML завершены локальные тесты. Для состоявшихся запусков сохранены
фактический argv, exit code, длительность и неизменность исходных входов;
для пропусков нет выдуманного кода завершения.
Сумма интервалов контроллеров, записанных в итоговом индексе, — **1156.9 s**. Это не точное полное wall time: сумма не включает три сохранённых результата прерванного candidate-extra, незафиксированный интервал инфраструктурного сбоя и отдельную перепроверку 41.2 s, указанную ниже. Предварительные smoke/exploratory contract заняли отдельно 98.2 s; unit/integration checks также учитываются отдельно.

| Сохранённый run directory | Фактические статусы | Re-audit CSV |
| --- | --- | ---: |
| `base-contract-final-81221` | `contract_valid`: 27, `expected_refusal`: 10, `solver_error`: 1 | 0 |
| `candidate-contract-81221` | `contract_valid`: 28, `expected_refusal`: 10 | 0 |
| `base-core-81221` | `expected_refusal`: 10, `full_result`: 13 | 13 |
| `candidate-core-81221` | `expected_refusal`: 10, `full_result`: 13 | 13 |
| `base-default-81221` | `full_result`: 3 | 3 |
| `candidate-default-81221` | `full_result`: 3 | 3 |
| `base-extra-81221` | `full_result`: 9, `solver_error`: 3 | 11 |
| `candidate-extra-81221` (partial, infrastructure interruption) | `full_result`: 3 | 3 |
| `candidate-extra-resume-81221-probability_columns_permuted_logloss` | `full_result`: 1 | 1 |
| `candidate-extra-resume-81221-future_inventory_mae` | `full_result`: 1 | 1 |
| `candidate-extra-resume-81221-grouped_logic_macro_f1` | `full_result`: 1 | 1 |
| `candidate-extra-resume-81221-high_cardinality_logloss` | `full_result`: 1 | 1 |
| `candidate-extra-resume-81221-tiny_minority_weighted_f1` | `solver_error`: 1 | 1 |
| `candidate-extra-resume-81221-impossible_group_split` | `solver_error`: 1 | 0 |
| `candidate-extra-resume-81221-markdown_literal_labels_accuracy` | `full_result`: 1 | 1 |
| `candidate-extra-resume-81221-known_linear_literal_ids` | `solver_error`: 1 | 0 |
| `candidate-extra-resume-81221-known_binary_permuted_features` | `full_result`: 1 | 1 |
| `base-wide_numeric_probe-stage2-81221` | `full_result`: 1 | 1 |
| `base-high_cardinality_probe-stage2-81221` | `full_result`: 1 | 1 |
| `base-censored_budget_probe-stage2-81221` | `skipped_cleanup_unavailable`: 1 | 0 |
| `candidate-wide_numeric_probe-stage2-81221` | `full_result`: 1 | 1 |
| `candidate-high_cardinality_probe-stage2-81221` | `full_result`: 1 | 1 |
| `candidate-censored_budget_probe-stage2-81221` | `skipped_cleanup_unavailable`: 1 | 0 |
| `candidate-known-linear-budget-recheck-81221` | `full_result`: 1 | 1 |

Имена в таблице относительны к игнорируемому `data/experiments/night_unseen/`.
Планы содержат `manifest.json`, `summary.json`, исходные task/data и audit JSON;
для состоявшихся исполнений сохранены результаты, обрезанные sanitized logs и
доступные raw predictions. Пропуск до запуска не объявляется исполнением.
`reaudit.json` проверяет сохранённые прогнозы без новых обучений. `not_scored`
для ожидаемого отказа или случая без CSV не объявляется пройденной ML-оценкой.

Сбой exec-service прервал первый candidate-extra после трёх записанных результатов.
Состояние исходного индекса сохранено в `stage1-index-preserved.json`, причина —
в `stage1_interruption.json`. У четвёртой попытки нет terminal receipt:
`exit_code=null`, исход неизвестен; ей не приписаны успешное завершение или код
ошибки. Девять отсутствовавших результатов выполнены в новых отдельных
`candidate-extra-resume-81221-<case>`; нижняя таблица объединяет их с исходными
тремя, без повторения уже сохранённых случаев. Вторая серия имеет отдельный
`stage2-index.json`.

Дополнительный независимый core audit проверил 26 полных результатов,
312 хешей артефактов (по 12 на результат), четыре staged CSV, checkpoint и
реальные full-data markers. Вторая реализация через прямые формулы regression,
confusion matrix, pairwise AUC и LogLoss пересчитала девять метрик/направлений и
per-class данные: совпадение с сохранённым аудитом до `1e-12`. Новых fit для
этого не было; оценки и targets не раскрывались. Метаданные сохранены в
`core_independent_check_81221.json`.

## Реальное обучение и выходные форматы

`full_result` требует независимого пересчёта метрики из CSV и реальных matching
checkpoint/refit-артефактов: положительное fit time, worker marker полного
train, совпадающие staged CSV и их включение в хешируемый manifest.
`valid_prior_fallback` означает отдельный target prior с `quality_verified=false`.

| Случай | Метрика | База | Candidate |
| --- | --- | --- | --- |
| `flow_rmse` | RMSE | `full_result` | `full_result` |
| `inventory_mae` | MAE | `full_result` | `full_result` |
| `fatigue_mse` | MSE | `full_result` | `full_result` |
| `binding_r2` | R2 | `full_result` | `full_result` |
| `logic_accuracy` | Accuracy | `full_result` | `full_result` |
| `rings_macro_f1` | Macro F1 | `full_result` | `full_result` |
| `cascade_weighted_f1` | Weighted F1 | `full_result` | `full_result` |
| `survival_binary_auc` | ROC AUC | `full_result` | `full_result` |
| `count_binary_logloss` | LogLoss | `full_result` | `full_result` |
| `checker_multiclass_logloss` | LogLoss | `full_result` | `full_result` |
| `checker_multiclass_auc` | ROC AUC | `full_result` | `full_result` |
| `reverse_positive_binary_auc` | ROC AUC | `full_result` | `full_result` |
| `numeric_all_missing_rmse` | RMSE | `full_result` | `full_result` |

Полноценные метрики candidate: `Accuracy`, `LogLoss`, `MAE`, `MSE`, `Macro F1`, `R2`, `RMSE`, `ROC AUC`, `Weighted F1`.

Точная схема, literal IDs и исходный порядок test проверены во всех сохранённых
CSV. Это включает бинарные/многоклассовые вероятности, явно положительный `NA`,
знаковые регрессионные targets и сохранение класса `01`. Дополнительные full
solver cases проверяют остальные варианты схем, времени, групп и literal labels:

| Дополнительный случай | База | Candidate |
| --- | --- | --- |
| `signed_flow_unicode_schema` | `full_result` | `full_result` |
| `missing_markers_inventory` | `full_result` | `full_result` |
| `literal_labels_accuracy` | `full_result` | `full_result` |
| `probability_columns_permuted_logloss` | `full_result` | `full_result` |
| `future_inventory_mae` | `full_result` | `full_result` |
| `grouped_logic_macro_f1` | `full_result` | `full_result` |
| `high_cardinality_logloss` | `full_result` | `full_result` |
| `tiny_minority_weighted_f1` | `solver_error` | `solver_error` |
| `impossible_group_split` | `solver_error` | `solver_error` |
| `markdown_literal_labels_accuracy` | `solver_error` | `full_result` |
| `known_linear_literal_ids` | `full_result` | `solver_error` |
| `known_binary_permuted_features` | `full_result` | `full_result` |

Результаты разных форматов внутри одного семейства не считаются независимыми
новыми датасетами. Численные audit quality-таблицы и per-class показатели
остаются в локальных summary; они не передавались разработчику модельной
политики. Разница случайных ML scores не используется
как доказательство correctness-исправления. Первая версия другого ИИ здесь
не квалифицировалась: требуется его точный frozen SHA.

### Ограниченный повтор budget-пробы

Первый `known_linear_literal_ids` candidate при бюджете 30 s дал `exit 1`,
без итогового CSV: seed получил внутренний timeout около 5.5 s, следующие
попытки — около 1.1–1.2 s; ни одной проверенной модели не осталось. В stderr
также сообщено, что emergency worker завершился ошибкой или исчерпал своё окно.
Эта ошибка сохранена в исходной матрице, не заменена более удачным повтором.

Отдельный заранее ограниченный recheck с теми же данными/seed и source
`3281a863`, бюджетом solver 60 s / external 120 s завершился `full_result`,
exit 0 за 41.2 s. Независимый re-audit подтвердил CSV и refit-артефакты.
Каталог: `candidate-known-linear-budget-recheck-81221`. Ни код, ни модельные
рецепты ради повтора не менялись. Это свидетельство чувствительности короткого
планирования к запуску worker, а не превышения платформенных трёх часов.

Запрос владельцу model/budget модулей: учитывать измеренный cold start и полный
минимальный цикл fit/predict/refit при выдаче коротких внутренних окон. Причина
отказа emergency worker без подробного terminal receipt остаётся неустановленной;
одна лишь общая строка stderr не доказывает конкретный механизм его сбоя.

### Два валидных случая за границей поддерживаемого holdout

`tiny_minority_weighted_f1` содержит единственный пример minority class;
стратифицированный split в `agent/splits.py:50` невозможен.
В `impossible_group_split` каждый класс находится в одной группе исходных
признаков, что отклоняет проверка `agent/splits.py:89–90`. Это известное
ограничение baseline solver, а не доказательство неправильной обученной модели.
В исходных baseline-запусках оба процесса вернули `exit 1 / solver_error`, хотя
на диске остался корректный CSV с verified `target_prior` и
`quality_verified=false`. Статус candidate показан выше по фактическому receipt;
наличие prior не превращает ошибочный exit в успешный full-model run.
В candidate для singleton-класса prior подтверждён, а для невозможного grouped
split CSV не создан: emergency worker сообщил failure либо исчерпание бюджета.
Причина этих двух вариантов не различима по доступному stderr.

Запрос владельцу solver: после успешной проверки `TaskSpec` обозначать только
доказанную невозможность holdout отдельной typed capability error, чтобы CLI
мог завершить проверенное emergency-восстановление в том же deadline.
Общий `except ValueError` недопустим: он скрыл бы некорректные входы. Группы
дублей нельзя молча разрывать ради score. Чужие `agent/splits.py` и
`solution.py` в этой ветке не менялись; локальный запрос с точными argv сохранён
в `reproductions/difficult_split_integration_request.md`.

## Некорректные входы, default CLI и probes

| Негативный случай | База | Candidate |
| --- | --- | --- |
| `missing_metric` | `expected_refusal` | `expected_refusal` |
| `conflicting_metric` | `expected_refusal` | `expected_refusal` |
| `ambiguous_positive` | `expected_refusal` | `expected_refusal` |
| `ambiguous_probability_mapping` | `expected_refusal` | `expected_refusal` |
| `duplicate_test_ids` | `expected_refusal` | `expected_refusal` |
| `missing_train_target` | `expected_refusal` | `expected_refusal` |
| `test_schema_mismatch` | `expected_refusal` | `expected_refusal` |
| `malformed_csv` | `expected_refusal` | `expected_refusal` |
| `missing_test_id` | `expected_refusal` | `expected_refusal` |
| `task_type_conflict` | `expected_refusal` | `expected_refusal` |

Содержательный отказ определяется по структурированной/терминальной диагностике,
а не по наличию traceback или имени каталога. Ошибка зависимостей не засчитывается
как ожидаемый отказ. Фактические ненулевые exit codes сохранены без подмены.

Default запускает ровно два позиционных пути: Python, `solution.py`, `task.md`,
без CLI-флагов; в manifest пути абсолютные, cwd — каталог задачи. Без API settings
это проверка default auto→offline, не работа Qwen/LLM. Flat-root архив отдельно
проверен существующими tests во временных каталогах, не заменой delivery ZIP.

| Default case | База | Candidate |
| --- | --- | --- |
| `flow_rmse` | `full_result` (exit 0) | `full_result` (exit 0) |
| `logic_accuracy` | `full_result` (exit 0) | `full_result` (exit 0) |
| `checker_multiclass_logloss` | `full_result` (exit 0) | `full_result` (exit 0) |

Наличие early CSV не отменяет внешний timeout. Любое `external_timeout` —
цензурированное наблюдение стенда, а не доказанное нарушение трёх часов платформы.

| Probe | Train / test | Solver / external, s | База | Candidate |
| --- | --- | --- | --- | --- |
| `wide_numeric_probe` | 1200 / 240 | 20 / 90 | `full_result` | `full_result` |
| `high_cardinality_probe` | 1600 / 320 | 20 / 90 | `full_result` | `full_result` |
| `censored_budget_probe` | 1600 / 320 | 15 / 8 | `skipped_cleanup_unavailable` | `skipped_cleanup_unavailable` |

Оба плана с внешним пределом 8 s имеют
`skipped_cleanup_unavailable`: в root PID namespace этого окружения нельзя
надёжно инвентаризировать собственных потомков для намеренной остановки.
Solver для них не запускался, fit и фактического timeout не было. Это
проверенный предварительный пропуск, а не успешная проверка deadline/recovery.

RSS/CPU telemetry в этом окружении недоступна из-за PID namespace; значения
не выдумывались. Memory monitor — best effort, не жёсткий OS memory limit.
Даже при доступном измерении sampled CPU может не учитывать уже завершившихся
детей. Это малые probes, а не проверка на максимальных 32 GiB / трёх часах.
Строгий CSV и загрузка TaskSpec в standalone validator добавляют чтение train;
производительность на больших настоящих данных этим набором не доказана.

## Проверки и обязательная интеграция перед merge

89 профильных runtime/case tests и первоначальные 19 tests runner прошли.
После доработки остановки процессов оба новых модуля стенда прошли 32 tests,
один real detached-child proof пропущен здесь из-за PID inventory; его проверяет native CI. Полный `dev_check.py`
выполнил 646 tests за 581.415 s: **1 error, 5 skips**. App tests отдельно:
227 за 77.967 s, 9 skips, OK. Полный набор не объявляется зелёным.

Единственная ошибка — compatibility interaction этого PR с legacy V2:
`test_source_mutation_after_good_experiment_cannot_finish_completed`.
Более строгий validator правильно отклоняет повреждённый исходный train,
после чего V2 скрывает integrity ValueError за общим RuntimeError step limit.
База проходит изолированный тест; candidate воспроизводит единственный error.
Требуется правка владельца `agent/research.py`; по разделению файлов она здесь
не применялась. Строгая validation не ослаблялась ради зелёного теста.

```diff
                     else:
                         result["selection"] = {"accepted": False, "verified": False,
                                                "reason": "execution_failed"}
+                    # Original input changes abort the task, even when a
+                    # candidate was rejected or its execution failed.
+                    state.check_source()
                     # The task-level file always contains accepted bytes, even if
                     # generated code attempted to write through an absolute path.
                     if state.backup.is_file():
                         result["previous_output_restored"] = state.recover()
```

The existing outer handler at lines 508–513 then records status `failed`, saves the report, restores the accepted bytes with `state.recover()`, and re-raises the original `ValueError`. No signature change is needed: the actual method is `check_source(self) -> None`.


Команда до/после интеграции:

```bash
python -m unittest tests.test_research.ResearchTests.test_source_mutation_after_good_experiment_cannot_finish_completed -v
python dev_check.py
```

Следует пропускать source-integrity исключение мимо per-experiment rejection
handler: `state.check_source()` после selection if/else перед `recover()`.
Проверка должна восстановить ожидаемый ValueError и сохранённый accepted CSV.
Прохождение предложенной, но не применённой правки не заявляется.

Native Linux/Windows CI публичного test-only SHA `78db10f` после Windows-fix
выполнил по 646 тестов; на обеих платформах остался только указанный V2 error.
Первоначальные Windows-проблемы новых тестов UTF-8/path comparison исправлены,
но suite всё ещё не зелёный. Run:
https://github.com/NolReaction/alfa-automl-agent/actions/runs/39527950368 .

Для runner SHA `3281a863` запущен новый native CI:
https://github.com/NolReaction/alfa-automl-agent/actions/runs/39529777424 .
Оба native job завершены: Linux — 651 tests за 495.188 s, 1 error / 2 skips;
Windows — 651 tests за 661.333 s, 1 error / 1 skip. На обеих платформах
единственная ошибка — тот же legacy V2 source-integrity test. Новые tests
runner прошли; реальный timeout worker с отдельной session подтверждён в
Linux CI (`test_timeout_stops_a_worker_in_a_new_session_when_inventory_is_available`).
На Windows этот POSIX-only proof закономерно skipped, а проверка retained
orphans/PID reuse прошла. Suite не зелёный: перед merge нужна указанная
интеграционная правка владельца V2. App/упаковочные шаги этих CI jobs после
ошибки core не выполнялись; локальные 227 app tests и временные archive tests
проверены отдельно. Позднейший коммит отчёта меняет только Markdown;
результаты native CI относятся к точному исполняемому Python SHA выше.

## Команды повторения и будущий frozen checkout

Из каталога аудитора с установленным requirements. Каждый output directory
должен быть новым; `--run` выполняется один раз, re-audit не обучает модель.
Для пары подставить соответственно frozen base или candidate solver root.

```bash
python qualify_unseen_tasks.py --solver-root /absolute/frozen/solver --out-dir data/experiments/night_unseen/repro-contract-001 --modes contract --seed 81221 --timeout 30 --total-seconds 300
python qualify_unseen_tasks.py --run data/experiments/night_unseen/repro-contract-001

python qualify_unseen_tasks.py --solver-root /absolute/frozen/solver --out-dir data/experiments/night_unseen/repro-core-001 --modes offline --seed 81221 --solver-seconds 30 --timeout 75 --total-seconds 1800 --cases flow_rmse inventory_mae fatigue_mse binding_r2 logic_accuracy rings_macro_f1 cascade_weighted_f1 survival_binary_auc count_binary_logloss checker_multiclass_logloss checker_multiclass_auc reverse_positive_binary_auc numeric_all_missing_rmse missing_metric conflicting_metric ambiguous_positive ambiguous_probability_mapping duplicate_test_ids missing_train_target test_schema_mismatch malformed_csv missing_test_id task_type_conflict
python qualify_unseen_tasks.py --run data/experiments/night_unseen/repro-core-001
python qualify_unseen_tasks.py --re-audit data/experiments/night_unseen/repro-core-001

python qualify_unseen_tasks.py --solver-root /absolute/frozen/solver --out-dir data/experiments/night_unseen/repro-default-001 --modes default --seed 81221 --timeout 90 --total-seconds 300 --cases flow_rmse logic_accuracy checker_multiclass_logloss
python qualify_unseen_tasks.py --run data/experiments/night_unseen/repro-default-001
python qualify_unseen_tasks.py --re-audit data/experiments/night_unseen/repro-default-001
```

Это команды повторения, а не автоматическое разрешение на новую бесконечную
серию. Точный интерпретатор задаётся `--python /absolute/venv/bin/python`, если
он отличается от Python аудитора; не resolve venv symlink до base interpreter.
Для будущего первого ИИ выбрать чистый checkout его зафиксированного SHA,
подставить путь вместо `/absolute/frozen/solver` и создать новые каталоги.
До preflight сверить `git rev-parse HEAD` и `git status --short`; manifest
закрепит SHA, source fingerprint и версии. Нынешний PR не ждёт его ветки.

Runner SHA `3281a863` дополнительно отслеживает собственных потомков и их
идентичность; один process group недостаточен для worker с новой session.
Это polling-проверка для обычных собственных процессов стенда, не гарантия
против произвольного adversarial process escape. Без доступной инвентаризации
намеренные короткие остановки пропускаются до запуска; после принудительной
остановки с неподтверждённой очисткой серия должна прерваться. Состоявшиеся
ранние normal-exit результаты первой серии сохранены отдельно от этого
исправления инфраструктуры.
Synthetic qualification не является оценкой Boosters и не устанавливает
причину прежних нулевых platform scores без worker logs. Реальных конкурсных
API-запросов: 0; платформенных отправок: 0; автоматического merge: 0.
