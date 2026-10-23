# Раунд 5, Мурад: готовность данных для OOF и скрытый сдвиг

Итог: инструмент `experiment_round5_data_readiness.py` строит полную таблицу сдвига TRAIN → TEST для обеих
публичных задач (62 и 42 признака, одна строка на признак) и отдельно проверяет переданный fit на совместимость с
разбиениями действующего OOF-кандидата. Модели не обучались, subprocess для обучения и API-запросов не было.
Рост конкурсной метрики этим инструментом не заявляется.

- Текущий профиль скрывает сдвиг: у регрессии 34 числовых признака с KS ≥ 0.1 не попадают в 20 записей
  `quality.numeric_shift`, у 9 медиана не изменилась, но KS ≥ 0.1. У классификации скрыт один такой признак.
- Полный публичный classification TRAIN — `structurally_compatible`. Блокеров разбиения нет; самый слабый класс
  Worms: 49 строк в A и 16 в B при пороге 5.
- Меньший fit может стать `ineligible`: на стратифицированных 20% TRAIN в B остаётся 4 строки Worms
  (`selection_insufficient_class_support`). Совместимость полного TRAIN не переносится на host INNER.
- Без обучения неизвестны прирост Macro F1, class-drop guard, допуск по времени, полный refit и память.

## База и окружение

| Параметр | Значение |
| --- | --- |
| База | `66e9481cb61cb83cfecab53be6984d7d2bf829b3` (совпадает с `origin/main` на 23.10.2026, разницы нет) |
| Ветка | `work/round5-murad-data-readiness`, worktree `../alfa-round5-murad` |
| Коммит отчётов | `ed56975` (инструмент `ff0f83b` + документы) |
| Python | 3.12.0, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, psutil 6.1.1, Windows 11 |
| Новые зависимости | нет |
| Файлы | пять новых: `experiment_round5_data_readiness.py`, `tests/test_round5_data_readiness.py`, `docs/round5_data_readiness.md`, `docs/round5_murad_report.md`, `docs/round5_data_readiness_integration_request.md` |

`agent/*`, общий host, CLI, splits/guards, `app/profiling.py`, `analyze_data.py`, зависимости, UI и сборка не
менялись. Используемые функции импортируются; две из них приватные — см. integration request.

## Команды (проверены, Windows PowerShell)

Из worktree, данные в основной копии:

```powershell
$py = "..\alfa-automl-agent\.venv\Scripts\python.exe"
$cls = "..\alfa-automl-agent\data\classification"
& $py .\experiment_round5_data_readiness.py shift --data-dir ..\alfa-automl-agent\data --task regression --output-dir ..\round5-reports\shift-regression
& $py .\experiment_round5_data_readiness.py shift --data-dir ..\alfa-automl-agent\data --task classification --output-dir ..\round5-reports\shift-classification
& $py .\experiment_round5_data_readiness.py fit --task-md "$cls\task.md" --dictionary "$cls\multiclass\participants\data_dictionary.csv" --fit-csv "$cls\multiclass\participants\train.csv" --output-dir ..\round5-reports\fit-classification-train
& $py -m unittest tests.test_round5_data_readiness tests.test_round4_feature_passport tests.test_analyze_data -v
```

Для fit-подвыборок с полным списком классов добавлялся
`--classes "Analysis,Backdoor,DoS,Exploits,Fuzzers,Generic,Normal,Reconnaissance,Shellcode,Worms"`.

## Сдвиг TRAIN → TEST

| | regression | classification |
| --- | --- | --- |
| Статус | `complete` | `complete` |
| TRAIN / TEST строк | 53 308 / 22 449 | 87 670 / 41 166 |
| Строк таблицы (признаков) | 62 | 42 |
| Числовых: сдвиг медиан посчитан / в профиле | 59 / 20 | 39 / 20 |
| KS ≥ 0.1 / ≥ 0.3 | 54 / 47 | 2 / 0 |
| KS ≥ 0.1 вне 20 записей профиля | 34 | 1 (`ct_dst_src_ltm`) |
| KS ≥ 0.1 при неизменной медиане | 9 | 1 (`ct_dst_src_ltm`) |
| Признаков с новыми категориями | 0 | 1 (`state`: 2 значения, 3 строки TEST из 41 166) |
| Пропуски изменились ≥ 1 п.п. (пусто или пробелы) / в профиле | 6 / 6 | 0 / 0 |
| Константных / с нечисловыми значениями TRAIN | 0 / 0 | 0 / 0 |
| `dataset_fingerprint` | `56d79494c81d27175bf15143c49b5755639ca8df51a77c718c6683b4365d3098` | `429f1ed681447fb30a2ba756c7c0dd01cccf104de61d82d2d43de2af2b62bd67` |
| `features_sha256` | `d0c033b37f874463aa14be8cdb77d3b1b02d59cdb75c4ee23e24eafe80a4889c` | `e2098f3e73fedbbc30519585f61ef66fd98d12f8f5bfe230ba558a13e0f5b804` |
| `content_sha256` | `4b4f32657ca3a0b38e5ece39151713309069242b6eab98a04533d214dfb6d6d4` | `d90527aa3a4acf17ed816fb9235922af661fc9ef0ccafd4bb4619d5ff64ad107` |

Контрольная проверка: 62 и 42 строки совпали с ожидаемым числом признаков. Это результат прогона, в коде число
не задано. Все 20 записей `quality.numeric_shift` в обеих задачах совпали с полным расчётом тем же
`_numeric_shift`. Fingerprint совпал с паспортом раунда 4. Статус `complete`: TEST-пропуски, не попавшие в
`missing_changes`, имеют причину `change_below_1pp_not_reported`, а значения analyze_data известны для всех
признаков.

### Что скрывал профиль

Профиль ранжирует сдвиг по медиане и оставляет 20 признаков. У регрессии 20-е место —
`days_above_27C` с `median_shift_score` 0.62. Поэтому признаки с сильным изменением формы, но меньшим сдвигом
медианы не видны:

| Признак (regression) | KS | Медиана TRAIN → TEST | median_shift_score |
| --- | --- | --- | --- |
| `december_avg_temp_c` | 0.869 | 5.26 → 3.52 | 0.375 |
| `may_avg_temp_c` | 0.713 | 17.78 → 17.11 | 0.524 |
| `days_below_minus18C` | 0.645 | 0 → 1 | 0.318 |
| `november_avg_temp_c` | 0.608 | 7.39 → 9.87 | 0.536 |
| `november_max_temp_c` | 0.583 | 21.11 → 22.22 | 0.400 |
| `heating_degree_days_c` | 0.528 | 2477.8 → 2362.2 | 0.266 |

`december_avg_temp_c` — второй по KS признак задачи после `Year_Factor` (KS 1.0). При неизменной медиане
сдвинуты, например, `october_min_temp_c` (KS 0.427), `july_max_temp_c` (0.362), `july_min_temp_c` (0.348),
`january_max_temp_c` (0.338). Изменение доли пропусков: у `direction_max_wind_speed`, `direction_peak_wind_speed`,
`max_wind_speed` 0.49 → 0.66, у `energy_star_rating` 0.42 → 0.20, у `days_with_fog` 0.62 → 0.57, у `year_built`
0.032 → 0.005.

У классификации сдвиг слабый: наибольший KS у `sinpkt` (0.111, в профиле). Вне профиля — `ct_dst_src_ltm`:
KS 0.104 при медиане 3 → 3. Признаки не удалялись и не преобразовывались.

## Fit-only проверка OOF

Кандидат — `make_recipe(fixed_incumbent_recipe(), classes, variant="oof")`. На публичных классах recipe hash
`cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc` совпадает с замороженным финалистом
раунда 4. Guards — `DEFAULT_GUARD_OPTIONS` без изменений.

| fit | Строк / групп | A / B | Мин. класс A / B | Решение | Причина |
| --- | --- | --- | --- | --- | --- |
| публичный TRAIN | 87 670 / 54 228 | 70 155 / 17 515 | 49 / 16 (Worms) | `structurally_compatible` | — |
| публичный outer grouped fit (имитация INNER) | 70 155 / 43 384 | 55 945 / 14 210 | 41 / 8 (Worms) | `structurally_compatible` | — |
| публичные стратифицированные 20% | 17 533 / 13 467 | 13 928 / 3 605 | 9 / 4 (Worms) | `ineligible` | `selection_insufficient_class_support` |
| синтетика: совместимый | 600 / 600 | 480 / 120 | 81 / 19 | `structurally_compatible` | — |
| синтетика: дубли с разными labels | 640 / 600 | 509 / 131 | 77 / 23 | `structurally_compatible` | — |
| синтетика: редкий класс (6 строк) | 506 / 506 | 405 / 101 | 4 / 2 | `ineligible` | `tuning_insufficient_class_support` |
| синтетика: объявлен отсутствующий класс | 600 / 600 | 480 / 120 | 0 / 0 (delta) | `ineligible` | `tuning_missing_class`; `fit_missing_class` у всех child-fit |
| синтетика: класс в одной группе | 510 / 501 | — | — | `ineligible` | `split_rows`: «один из классов встречается только в одной группе признаков» |

Полный TRAIN: A/B 70 155 / 17 515 — те же размеры, что в full-refit раунда 4. Три fold: fit/held
47 021 / 23 134, 46 421 / 23 734, 46 868 / 23 287 строк. Пересечений групп A/B и fit/held нет, каждая строка A
попала в held ровно один раз. Child-fit по размерам проходят: 87 670 ≤ 100 000 fit-строк, 10 ≤ 50 классов.
Плановых физических fits 10 при cap 10.

Дубли raw-признаков на полном TRAIN: 5 816 групп с копиями, 39 258 строк в них, 33 442 лишние копии. У 1 458 групп
разные labels (14 829 строк; больше всего у Exploits 6 503 и DoS 4 838). По действующей политике это одна группа,
не ошибка.

Outer grouped fit — первая часть канонического `split_rows` на TRAIN; это лишь приближение INNER. Его B содержит
8 строк Worms, запас над порогом 5 — три строки. Fit-подвыборка 20% уже не проходит gate. Поэтому допуск нужно
проверять на настоящем INNER host, а не на полном TRAIN.

### Что блокирует и что неизвестно

На предоставленном полном TRAIN структурных блокеров нет. Остаются `unknown` без фиктивного `passed`:

- прирост Macro F1 на B ≥ 0.01 и падение F1 класса ≤ 0.02 — нужны OOF-вероятности;
- допуск по времени `4·probe·1.5 + max(120, 2·probe) ≤ 600` — нужен измеренный identity probe;
- успешный полный refit, caps predict-таблицы (≤ 500 000 строк, ≤ 10 000 000 ячеек), RSS ≤ 6 GiB;
- совпадение проверенного fit с будущим host INNER.

## Время, размеры, память

| Прогон | `elapsed_seconds` | Wall с запуском Python | Пик памяти процесса | JSON / CSV / MD, байт |
| --- | --- | --- | --- | --- |
| shift regression | 17.4 с | 21.1 с | 221 MiB | 97 423 / 28 760 / 6 247 |
| shift classification | 21.6 с | 25.5 с | 280 MiB | 65 192 / 18 972 / 4 506 |
| fit публичный TRAIN | 41.8 с | 45.6 с | 292 MiB | 22 189 / 2 417 / 6 921 |
| fit outer grouped fit | 32.9 с | 36.6 с | 260 MiB | 21 937 / 2 404 / 6 681 |
| fit 20% | 9.8 с | 13.5 с | 169 MiB | 21 886 / 2 293 / 6 625 |
| fit синтетика (5 случаев) | 0.1–0.5 с | 3.6–4.2 с | 138 MiB | 10 780–16 888 / 85–774 / 5 780–6 288 |

Память — `psutil` peak working set всего процесса Python за запуск, включая импорты; tree RSS не измерялся. По
cProfile больше 90% fit-аудита полного TRAIN занимает `StratifiedGroupKFold` внутри канонических
`coverage`/`split_rows`. Тот же расчёт делает и сам кандидат внутри своего trial; это стоит учитывать в бюджете OOF.

## Воспроизводимость и сохранность входов

- Повторный полный прогон в другую папку (`round5-reports-repeat`) дал те же `content_sha256`, `config_sha256`,
  `inputs` и хеши файлов во всех 10 отчётах.
- SHA-256 всех входных файлов (публичные `regression`, `classification` и синтетические случаи) до и после двух
  прогонов совпали.
- В отчётах нет ID, построчных labels, индексов разбиений, ключей групп, примеров категорий и путей машины.

## Тесты

`tests/test_round5_data_readiness.py`, 15 тестов; вместе с `test_round4_feature_passport` и `test_analyze_data` —
30, все проходят. Обязательные проверки задания:

| Требование | Тест |
| --- | --- |
| 1. TEST изменён или удалён — fit-режим не меняется | `test_test_changes_or_deletion_do_not_affect_fit_mode`, `test_pure_audit_opens_no_file` |
| 2. Редкий или отсутствующий класс не становится допуском | `test_rare_or_absent_class_is_never_admitted` |
| 3. Raw-feature группы не пересекают границы | `test_duplicate_groups_with_conflicting_labels_never_cross_a_boundary`, `test_a_leaking_partition_is_detected_not_trusted` |
| 4. Неверная схема, labels, class order и размеры — понятный отказ | `test_wrong_schema_labels_class_order_and_size_give_clear_refusals`, `test_class_order_comes_from_the_declaration_or_the_command_line` |
| 5. Больше 20 сдвинутых числовых признаков сохраняются | `test_more_than_twenty_shifted_numeric_features_all_stay_in_the_table` |
| 6. Копия в другой папке — те же хеши | `test_copies_in_another_folder_keep_content_hashes` |
| 7. Входы побайтово те же | `test_inputs_stay_byte_identical_and_outputs_hold_no_rows` |
| 8. Нет обучения, subprocess для обучения и сети | `test_no_training_no_training_subprocess_and_no_network` |

Дополнительно: разные определения пропусков, std и новых категорий не смешиваются, у каждого `null` есть причина;
полный `analyze()`, утечки и корреляции не вызываются; расхождение с профилем останавливает экспорт; папка вывода
внутри входов запрещена, прерванная запись помечается `status: error`. Полный набор `tests` не запускался: в
общем коде изменений нет.

## Архив

`round5-reports.zip` (вне репозитория) содержит десять папок отчётов и `SHA256SUMS.txt`. Исходных данных и
подвыборок в нём нет; подвыборки для fit-случаев строились во временной папке.

## Ограничения

- `_numeric_shift` и `_inputs` — приватные функции; при их изменении инструмент нужно сверить заново.
- Для TEST нечисловые и ±inf не считает ни один источник; доля пустых ячеек TEST известна только из
  `missing_changes`.
- `structurally_compatible` означает, что первый OOF-fit был бы начат, а не что модель будет принята.
- Публичные подвыборки (outer grouped fit, 20%) — иллюстрация поддержки, не настоящий host INNER.
