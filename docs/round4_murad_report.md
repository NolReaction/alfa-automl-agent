# Раунд 4, Мурад: отчёт по паспорту признаков

Итог: инструмент воспроизводимо создаёт `passport.json`, `features.csv` и `README.md` для обеих публичных задач.
Входные файлы не меняются, target, ID и сырые значения в экспорт не попадают. Новых ошибок в `app/profiling.py`
не найдено. Найдено ограничение профайлера, которое скрывает часть сдвинутых признаков (раздел «Что найдено»).
Это developer-инструмент: в runtime и в конкурсный архив он не подключается. Метрика качества и физические fits
здесь `not_applicable`: модели не обучались.

## База и окружение

| Параметр | Значение |
| --- | --- |
| База | `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb` (совпадает с `origin/main` на 23.10.2026) |
| Ветка | `work/round4-murad-passport-v1`, worktree `../alfa-round4-murad` |
| Python | 3.12.0, numpy 1.26.4, pandas 2.2.3, Windows 11 |
| Новые зависимости | нет |
| Изменённые файлы | ровно четыре новых: `experiment_round4_feature_passport.py`, `tests/test_round4_feature_passport.py`, `docs/round4_feature_passport.md`, `docs/round4_murad_report.md` |

Существующие `analyze_data.py`, `app/profiling.py`, `agent/*`, `research_lab/*`, `solution.py`, UI, CI и
зависимости не менялись.

## Команды (проверены)

Из worktree, данные подготовлены `prepare_data.py` в основной копии:

```powershell
..\alfa-automl-agent\.venv\Scripts\python.exe .\experiment_round4_feature_passport.py --data-dir ..\alfa-automl-agent\data --task regression --output-dir ..\alfa-passport\regression
..\alfa-automl-agent\.venv\Scripts\python.exe .\experiment_round4_feature_passport.py --data-dir ..\alfa-automl-agent\data --task classification --output-dir ..\alfa-passport\classification
..\alfa-automl-agent\.venv\Scripts\python.exe -m unittest tests.test_round4_feature_passport tests.test_analyze_data -v
..\alfa-automl-agent\.venv\Scripts\python.exe -m unittest discover -s app/tests -p test_profiling.py -v
```

Идентификаторы `--task` совпали с примером задания: `discover()` возвращает `regression` и `classification`.

## Результаты экспорта

| | regression | classification |
| --- | --- | --- |
| Статус | `partial` | `partial` |
| TRAIN / TEST строк | 53 308 / 22 449 | 87 670 / 41 166 |
| Признаков (без target и ID) | 62 | 42 |
| `dataset_fingerprint` | `56d79494c81d27175bf15143c49b5755639ca8df51a77c718c6683b4365d3098` | `429f1ed681447fb30a2ba756c7c0dd01cccf104de61d82d2d43de2af2b62bd67` |
| `content_sha256` | `448e58f6d4aa6c493232f3932f85e7b76f5c27a3696f97393184757ac9b1ee17` | `d00b9ef653c09600d7aee96ad9885ab53fa9ab97292d386f64a5de13c8682bfc` |
| `features_sha256` | `6aa8a4b456689f33c7c8c2e63194d3d6a6cd50dcf781fec928cb697ee8009494` | `ea66b1bf494735cc18b7c8075ecf653cb5cbf470532ade215073f919cca5b000` |
| Время экспорта (`elapsed_seconds`) | 15.2 с | 13.4 с |
| Повторный запуск, wall с запуском Python | 13.3 с | 12.2 с |
| `passport.json` / `features.csv` / `README.md`, байт | 59 523 / 19 807 / 5 020 | 39 740 / 11 772 / 4 673 |
| Пик памяти | `unknown`: не измерялся | `unknown`: не измерялся |

Статус `partial` в обеих задачах из-за одного источника: `quality.numeric_shift` заполнен до лимита 20,
поэтому у 39 признаков регрессии и 19 признаков классификации `shift_*` = `null` с причиной `outside_source_cap`.
Остальные списки не усечены: `new_categories` вернул 0 и 1 запись, `missing_changes` — 6 и 0.

Повторный экспорт в другую папку дал те же `dataset_fingerprint`, `features_sha256`, `content_sha256` и хеши
`features.csv`/`README.md`. SHA-256 всех десяти входных файлов до и после экспортов совпали. В трёх файлах
каждой задачи нет имён target (`site_eui`, `attack_cat`) и ID (`id`), нет путей пользователя.

## Сверка с `analyze_data.py`

Разовая сверка на тех же данных: `analyze_data.analyze()` против полей паспорта, по всем признакам.

| Поле паспорта | Поле `analyze_data` | regression: сравнено / расхождений | classification | Семантика |
| --- | --- | --- | --- | --- |
| `train_missing_count` | `missing_train × train_rows` | 62 / 0 | 42 / 0 | `analyze_data` обрезает пробелы, профайлер — нет; в публичных данных пробельных ячеек нет |
| `train_nonfinite_count` | `(unparseable + infinite) × train_rows` | 59 / 0 | 39 / 0 | совпадает |
| `train_unique_count` | `n_unique` | 62 / 0 | 42 / 0 | профайлер считает различные строки, анализатор — числа; в этих данных одинаково |
| `train_min`, `train_max`, `train_mean`, `train_median` | `min`, `max`, `mean`, `median` | 62 / 0 | 42 / 0 | совпадает |
| `train_std` | `std` | 62 / 59 | 42 / 39 | профайлер: `np.std`, ddof=0; анализатор: pandas, ddof=1. После поправки √(n/(n−1)) разница 0 или ~1e−14 |
| `test_missing_percent` | `missing_test × 100` | 6 / 0 | 0 / 0 | сравнивались только признаки, где профайлер сообщает TEST |
| `new_category_count` | `unseen_test_share` | — | — | разные величины: число различных новых значений против доли строк TEST. `state`: 2 значения, 3 строки из 41 166 |
| `is_constant` | находки «константа» | 0 и 0 | 0 и 0 | константных признаков нет в обоих источниках |

Расхождение `train_std` — разное определение, а не ошибка. Числа паспорта не подгонялись.

Ранжирование сдвига отличается сильнее. `shift_score` профайлера смотрит только на медианы, KS анализатора —
на всё распределение. Пересечение первых 10 признаков: 5 из 10 в регрессии и 1 из 10 в классификации.

## Что найдено

**Ошибок в профайлере не найдено.** Найдены два ограничения, которые важно знать при чтении паспорта.

1. **Лимит `numeric_shift` = 20 скрывает заметный сдвиг.** В классификации 5 из 10 признаков с наибольшим KS
   (`ct_dst_src_ltm`, `ct_dst_sport_ltm`, `ct_src_dport_ltm`, `ct_src_ltm`, `ct_dst_ltm`) не попали в список
   профайлера, в регрессии 2 из 10 (`december_avg_temp_c`, `may_avg_temp_c`). Паспорт честно ставит им `null`
   `outside_source_cap`, а не 0.
2. **Медианный `shift_score` не видит изменения формы распределения.** У `sinpkt`, первого по KS в
   классификации, `shift_score` = 0.0046.

Воспроизведение: экспорт `classification` командой выше, затем
`python -c "import analyze_data; r, p = analyze_data.analyze(__import__('pathlib').Path('../alfa-automl-agent/data/classification')); print(p.sort_values('ks_train_test', ascending=False).feature.head(10).tolist())"`
и сравнение с `null_reasons.shift_score` в `passport.json`.

Возможный `integration_request` для капитана (сам не делал, `app/profiling.py` не менял): либо отдавать
`numeric_shift` без обрезки до 20 (число признаков уже ограничено `MAX_COLUMNS = 512`), либо добавить в
`_numeric_shift` готовый KS. Тест: фикстура с 24 сдвинутыми числовыми признаками
(`test_source_caps_mark_partial_rows_and_the_output_stays_bounded`) должна получить `status: complete`.

Ещё ограничения источника, которые паспорт не досчитывает:

- нечисловые значения TEST не возвращаются;
- доля пропусков TEST известна, только если она отличается от TRAIN не меньше чем на 1 п.п. (не больше 10 признаков);
- `new_categories[].examples` содержат сырые значения, поэтому экспортируется только количество.

## Заметные факты и вопросы команде

**regression.** Новых категорий нет, константных и нечисловых значений нет. Сильнее всего сдвинуты погодные
признаки: `august_avg_temp_c` 23.7 → 26.2, shift_score 2.10. `Year_Factor` (5 значений) — медиана 4 → 6, TEST
лежит в более позднем периоде. Доля пропусков меняется у 6 признаков, сильнее всего у `energy_star_rating`:
41.7% → 20.0%; `max_wind_speed` и направления ветра — 49–51% → 65.8%.

**classification.** У `state` два значения TEST, которых нет в TRAIN; они встречаются в трёх строках. `swin` и `dwin` (8 и 4 значения) — медиана 0 → 255. Пропусков нет, константных и нечисловых значений
нет.

Вопросы в `README.md` каждой задачи:

1. Есть ли безопасный unknown fallback для новых значений `state`?
2. Как ведут себя сдвинутые признаки на TRAIN-срезах? Сдвиг сам по себе не повод удалять признак.
3. Не задают ли `Year_Factor`, `swin`, `dwin` период, группу или смесь подтипов, по которым стоит строить валидацию?
4. Как текущий transformer обрабатывает изменение доли пропусков у `energy_star_rating` и ветровых признаков?
5. Нужен ли расширенный лимит `numeric_shift`?

## Проверки

`tests/test_round4_feature_passport.py`, 9 тестов, все проходят. Существующие `tests.test_analyze_data`
(6 тестов) и `app/tests/test_profiling.py` (19 тестов, 1 пропущен: Windows junction) на этой ветке проходят без
изменений.

| Риск из задания | Тест |
| --- | --- |
| Входы до/после; нет `submission.csv`, модели, `.agent_runs` | `test_inputs_stay_byte_identical_and_only_three_files_are_written` |
| Перенос задачи сохраняет fingerprint и признаки; новый TRAIN target меняет только fingerprint | `test_fingerprint_is_portable_and_feature_data_ignore_the_train_target` |
| Target и ID с неожиданными именами и позициями; preview, описания, будущие поля профайлера не просачиваются | `test_roles_not_names_exclude_target_and_id_and_raw_values_never_leak` |
| Литеральный `NA`, `inf`, невалидное число; malformed CSV, нет столбца, нет файла — ошибка без вывода | `test_missing_semantics_follow_the_profiler_and_bad_inputs_fail_without_output` |
| Лимиты источника отражены в статусе и счётчиках; вывод ограничен | `test_source_caps_mark_partial_rows_and_the_output_stays_bounded` |
| Запятые, переносы, Markdown в именах; CSV через `csv.writer` | `test_names_with_commas_newlines_and_markdown_survive_csv_and_markdown` |
| Output внутри данных, на входном файле, на существующем файле — отказ до записи | `test_output_inside_data_or_on_a_file_is_refused_before_writing` |
| Повторный экспорт не трогает чужие файлы; сбой записи не оставляет `complete` | `test_reexport_keeps_foreign_files_and_a_failed_write_is_never_complete` |
| Фикстуры анализатора: константа, сдвиг, нечисловые значения, новые категории | `test_analyzer_fixtures_show_constants_shift_unparseable_and_new_categories` |

## Что можно подключать

- Сейчас: инструмент для разработчиков ИИ-1 и ИИ-2, чтобы быстро увидеть новые категории, сдвиг и пропуски.
- Не подключать: паспорт полного TRAIN не является INNER-контекстом для Qwen и не меняет runtime.
- Решение о расширении лимита `numeric_shift` — за капитаном.

Готовые паспорта обеих задач лежат вне репозитория архивом `alfa-round4-passports.zip`. В архиве два набора из
трёх файлов и `REPRODUCE.md` с командами; локальных путей и данных в нём нет.
