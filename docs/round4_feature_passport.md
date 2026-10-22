# Паспорт признаков (раунд 4)

`experiment_round4_feature_passport.py` — offline-инструмент разработчика. Он берёт готовый профиль
`app.profiling.DataCatalog` одной задачи и сохраняет его в три переносимых файла. Паспорт показывает новые
категории, сдвиг TRAIN → TEST, пропуски и плохие значения. Он не обучает модели, не выбирает модель и не
доказывает рост качества. В конкурсный runtime и в архив `build_submission.py` инструмент не входит.

## Запуск

```powershell
.\.venv\Scripts\python.exe .\experiment_round4_feature_passport.py --data-dir <папка с задачами> --task regression --output-dir <папка вне data>
.\.venv\Scripts\python.exe .\experiment_round4_feature_passport.py --data-dir <папка с задачами> --task classification --output-dir <другая папка вне data>
```

Проверенный вариант из отдельного worktree рядом с основной копией, где данные подготовлены `prepare_data.py`:

```powershell
..\alfa-automl-agent\.venv\Scripts\python.exe .\experiment_round4_feature_passport.py --data-dir ..\alfa-automl-agent\data --task regression --output-dir ..\alfa-passport\regression
```

- `--data-dir` — папка, внутри которой задачи лежат подпапками с `task.md` (так их находит `DataCatalog.discover()`).
- `--task` — `id` из `discover()`. Для публичных данных это `regression` и `classification`. При неизвестном
  значении CLI печатает список допустимых идентификаторов. Режима `all` нет: две команды подряд.
- `--output-dir` — обязательна и должна быть вне `--data-dir`. Путь внутри данных, путь к входному файлу и путь
  к существующему файлу отклоняются до чтения профиля и до любой записи.
- Импорт `app.profiling` не поднимает сервер и не открывает браузер. Нужны только `numpy` и `pandas` из
  `requirements.txt`.

Код выхода `0` — паспорт записан (`complete` или `partial`), `1` — ошибка, `2` — неверные аргументы.

## Что пишется

| Файл | Назначение |
| --- | --- |
| `passport.json` | основной машинный формат: метаданные, лимиты источника, строки признаков, хеши |
| `features.csv` | одна строка на признак, те же поля, что в `passport.json["features"]` |
| `README.md` | короткая сводка: заметные факты (до 10 строк на таблицу) и 3–5 вопросов команде |

Порядок записи: сначала удаляется старый `passport.json` этой папки, затем `features.csv` и `README.md`
пишутся через временный файл и `os.replace`, последним пишется `passport.json` с SHA-256 двух других файлов.
Если запись прервалась, `passport.json` либо отсутствует, либо имеет `status: "error"`; набор со статусом
`complete` при сбое не остаётся. Другие файлы в папке вывода не трогаются, `rmtree` не используется.

Ошибка чтения данных (malformed CSV, нет столбца, нет файла, неизвестная задача) прерывает работу до записи:
папка вывода не создаётся.

## Метаданные `passport.json`

| Поле | Значение |
| --- | --- |
| `schema_version` | `round4-feature-passport/1` |
| `task_id` | выбранный `id` из `discover()` |
| `scope` | всегда `developer_full_train_test_features` |
| `source` | `app.profiling.DataCatalog` |
| `source_commit` | `git rev-parse HEAD` каталога инструмента; `null` и предупреждение, если git недоступен |
| `dataset_fingerprint` | `DataCatalog.fingerprint()`: SHA-256 байтов `task.md` и четырёх входных файлов под стабильными именами, без путей |
| `train_rows`, `test_rows`, `feature_count` | `profile()["train_rows"]`, `["test_rows"]`, `["feature_count"]` |
| `status` | `complete`, `partial` или `error` (см. ниже) |
| `limitations` | постоянные ограничения источника |
| `warnings` | что именно усечено в этом наборе; отсутствие `source_commit` |
| `source_caps` | для каждого ограниченного списка: `cap`, `returned`, `may_be_truncated`, `features_outside_cap` |
| `field_sources` | поле паспорта → откуда оно взято в `DataCatalog` |
| `features` | строки признаков |
| `features_sha256` | SHA-256 канонического JSON списка `features` |
| `content_sha256` | SHA-256 канонического JSON всех полей выше |
| `files` | SHA-256 `features.csv` и `README.md` |
| `generated` | `generated_at` (UTC) и `elapsed_seconds` |

`generated` и `files` не входят в `content_sha256`. Временная отметка и локальные пути не влияют на сравнение,
а пути в паспорт не попадают совсем. Та же задача в другой папке даёт те же `dataset_fingerprint`,
`features_sha256` и `content_sha256`. Изменение TRAIN target меняет `dataset_fingerprint`, а `features` и
`features_sha256` остаются прежними.

## Поля признака

Target и технический ID исключаются по ролям словаря, которые возвращает профайлер (`profile()["target"]`,
`["id_column"]`, `columns[].role`), а не по именам.

| Поле паспорта | Публичный метод | Исходное поле | Фильтрация и смысл |
| --- | --- | --- | --- |
| `name`, `role`, `dtype` | `profile()` | `columns[].name`, `.role`, `.dtype` | роль и тип объявлены в `data_dictionary.csv`, профайлер приводит их к нижнему регистру |
| `train_missing_count`, `train_missing_percent` | `profile()` | `columns[].missing_count`, `.missing_percent` | TRAIN; пропуск — только пустая ячейка `""`, процент округлён профайлером до 4 знаков |
| `train_nonfinite_count` | `column()` | `distribution.invalid_count` | только `numeric`/`binary`: непустые значения TRAIN, которые не стали конечным числом (`inf`, `invalid`, `NA`) |
| `train_unique_count` | `profile()` | `columns[].unique_count` | число различных непустых строк в TRAIN (`"1"` и `"1.0"` различаются) |
| `is_constant` | `profile()` | `quality.constant_features` | не больше одного различного значения в TRAIN, пустая ячейка считается значением |
| `train_min` … `train_std` | `column()` | `stats.min/max/mean/median/std` | только `numeric`/`binary`, по конечным значениям TRAIN; `std` с ddof=0 (`np.std`) |
| `test_missing_percent` | `profile()` | `quality.missing_changes[].test_percent` | известна, только если доля пропусков TRAIN и TEST различается не меньше чем на 1 п.п. и признак попал в первые 10 |
| `new_category_count` | `profile()` | `quality.new_categories[].count` | только `category`: число различных непустых значений TEST, которых нет в TRAIN (не доля строк); список до 20 записей; сами значения не экспортируются |
| `shift_train_median`, `shift_test_median`, `shift_score` | `profile()` | `quality.numeric_shift[]` | только `numeric`/`binary`; `shift_score` = \|медиана TEST − медиана TRAIN\| / IQR TRAIN (при IQR = 0 — std TRAIN); список до 20 записей |
| `status` | — | — | `ok` или `partial`, если хотя бы одно поле пропало из-за лимита источника |
| `null_reasons` | — | — | для каждого `null` — причина; в CSV строка `поле=причина;...` |

Причины `null`:

| Причина | Значение |
| --- | --- |
| `not_numeric_dtype` | поле определено только для `numeric`/`binary` |
| `not_category_dtype` | поле определено только для `category` |
| `no_finite_train_values` | в TRAIN нет ни одного конечного числа |
| `outside_source_cap` | ограниченный список профайлера заполнен до лимита, признака в нём нет: значение неизвестно (может быть и 0) |
| `change_below_1pp_not_reported` | профайлер не сообщает долю пропусков TEST, если она отличается от TRAIN меньше чем на 1 п.п. |
| `no_finite_values_in_a_part` | профайлер не считает сдвиг: в TRAIN или TEST нет конечных чисел |
| `undefined_by_source` | профайлер вернул `null` (например, сдвиг изменившейся константы) |

Если признака нет в незаполненном списке `new_categories`, это настоящий `0`, а не `null`.

## Статусы

- `complete` — ни один ограниченный список источника не отрезал признаки.
- `partial` — хотя бы один список заполнен до лимита и у части признаков есть `outside_source_cap`.
  Сколько именно, видно в `source_caps` и `warnings`.
- `error` — запись набора прервалась. Такой `passport.json` содержит только `schema_version`, `task_id`,
  `scope`, `dataset_fingerprint`, `status` и `error`, остальным файлам папки доверять нельзя.

## Что не экспортируется

Паспорт собирается по явному списку `FIELD_SOURCES`. Новые поля профайлера в будущем не попадут в экспорт сами.
Никогда не выходят: `preview`, сырые строки и значения категорий (включая `new_categories[].examples`),
индивидуальные ID, `target_distribution`, `class_balance`, сводка `missing` с target и ID, `missing_cells`,
описания из словаря, текстовые `warnings` профайлера, метрика задачи, корреляции с target.

## Ограничения

- По отдельному признаку профайлер считает только TRAIN. Нечисловые значения TEST и полную долю пропусков TEST
  он не возвращает, а паспорт их не досчитывает.
- `shift_score` смотрит только на медианы. Изменение хвостов или формы распределения при той же медиане он не
  видит; KS из `analyze_data.py` ранжирует признаки иначе.
- Сдвиг и новые категории — повод исследовать признак, а не доказательство ухудшения предсказаний и не повод
  автоматически удалять признак.
- Паспорт описывает полный TRAIN и признаки TEST. Это не разрешённый INNER-контекст для Qwen: его строят
  отдельно на fit, переданном host.
