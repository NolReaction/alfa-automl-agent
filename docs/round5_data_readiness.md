# Раунд 5: готовность данных — сдвиг TRAIN → TEST и fit-only проверка OOF

`experiment_round5_data_readiness.py` — offline-инструмент разработчика с двумя независимыми режимами. Он не
обучает модели, не запускает subprocess для обучения и не делает API-запросов. В runtime, CLI агента и конкурсный
архив не подключается. `agent/*`, `app/profiling.py`, `analyze_data.py`, splits, guards и зависимости не меняются.

| Режим | Вопрос | Scope | Читает |
| --- | --- | --- | --- |
| `shift` | Какие изменения TRAIN → TEST скрывает текущий профиль? | `developer_full_train_vs_test_features` | задачу через `DataCatalog` и те же CSV через `analyze_data.Task` |
| `fit` | Позволяет ли переданный fit безопасно построить разбиения OOF-кандидата? | `explicit_fit_only_oof_structure` | только `task.md`, `data_dictionary.csv` и таблицу fit |

Shift-отчёт описывает полный TRAIN. Это не разрешённый INNER-контекст для Qwen. Fit-отчёт о полном TRAIN не
доказывает пригодность будущего host INNER: его нужно проверить на том fit, который передаст host.

## Команды (Windows PowerShell)

Из worktree; данные лежат в основной копии `..\alfa-automl-agent\data`. Папка вывода должна быть вне папок входов.

```powershell
$py = "..\alfa-automl-agent\.venv\Scripts\python.exe"
$cls = "..\alfa-automl-agent\data\classification"
& $py .\experiment_round5_data_readiness.py shift --data-dir ..\alfa-automl-agent\data --task regression --output-dir ..\round5-reports\shift-regression
& $py .\experiment_round5_data_readiness.py shift --data-dir ..\alfa-automl-agent\data --task classification --output-dir ..\round5-reports\shift-classification
& $py .\experiment_round5_data_readiness.py fit --task-md "$cls\task.md" --dictionary "$cls\multiclass\participants\data_dictionary.csv" --fit-csv "$cls\multiclass\participants\train.csv" --output-dir ..\round5-reports\fit-classification-train
& $py -m unittest tests.test_round5_data_readiness -v
```

Для fit другой выборки, например host INNER, порядок классов задайте явно: `--classes "Analysis,Backdoor,..."`
(или JSON-массив). Источник порядка: объявление в `task.md`, затем `--classes`, затем отсортированные метки fit.
В последнем случае отчёт предупреждает, что класс, которого нет в fit, так не виден. `--classes`, противоречащий
`task.md`, — отказ.

## Режим shift

Одна строка на каждый объявленный признак (`role` из словаря, не по имени столбца). Значения берутся только из
существующих расчётов; новых формул нет.

- `DataCatalog.profile()` и `column()` — TRAIN-сводки, константы, списки `quality`.
- `app.profiling._numeric_shift` — тот же сдвиг медиан, что в `quality.numeric_shift`, но для каждого числового
  признака. Все 20 записей профиля сверяются с полным расчётом; при расхождении экспорт останавливается.
- `analyze_data.profile_features` — пропуски с пробелами, нечисловые и ±inf, KS, выход за диапазон, доля строк с
  новыми категориями, TVD. Полный `analyze()` с утечками, корреляциями и target не вызывается.

Обе библиотеки читают CSV одинаково: `dtype=str`, `keep_default_na=False`, `utf-8-sig`. Fingerprint данных
проверяется до и после чтения, число строк и столбцов сверяется.

| Поле | Источник | Смысл |
| --- | --- | --- |
| `train_missing_percent_empty` / `test_missing_percent_empty` | DataCatalog | пустая ячейка, проценты; TEST — только из `missing_changes` (≥ 1 п.п., до 10 записей) |
| `train_missing_share_blank` / `test_missing_share_blank` | analyze_data | пустая ячейка или только пробелы, доли |
| `train_placeholder_share` | analyze_data | доля строк вроде `NA`, `null`, `?` (только сообщается: `NA` бывает настоящим значением) |
| `train_nonfinite_count` | DataCatalog | непустые нечисловые и ±inf вместе |
| `train_unparseable_share`, `train_infinite_share` | analyze_data | то же раздельно, доли |
| `test_nonfinite` | — | всегда `null`: ни один источник не считает это для TEST |
| `train_unique_nonempty` | DataCatalog | различные непустые строки |
| `train_unique_analyze` | analyze_data | числовые: различные конечные значения; категории: без пробелов по краям |
| `is_constant` | DataCatalog | не больше одного значения, пустая ячейка тоже значение |
| `train_std_ddof0` / `train_std_ddof1` | DataCatalog / analyze_data | `np.std` и pandas `std` |
| `train_median`, `test_median`, `median_shift_score` | `_numeric_shift` | \|Δ медианы\| в IQR TRAIN (std, если IQR = 0) |
| `median_shift_in_profile` | DataCatalog | входит ли признак в 20 записей `quality.numeric_shift` |
| `ks_train_test`, `test_out_of_range_share` | analyze_data | KS по конечным значениям; доля TEST вне [min, max] TRAIN |
| `new_category_count` | DataCatalog | различных новых непустых значений TEST как есть (список до 20) |
| `new_category_row_share` | analyze_data | доля всех строк TEST с новым значением после обрезки пробелов |
| `numeric_code_unseen_share` | analyze_data | доля конечных значений TEST, которых нет в TRAIN; только при ≤ 20 значениях в TRAIN |
| `tvd_train_test` | analyze_data | total variation distance распределений категорий |

У каждого `null` в `null_reasons` есть причина: `not_numeric_dtype`, `not_category_dtype`,
`no_finite_values_in_a_part`, `outside_source_cap`, `change_below_1pp_not_reported`, `not_provided_by_sources`,
`more_than_20_train_values`, `zero_train_spread_with_changed_median`, `no_finite_train_values`,
`no_nonempty_train_values`, `no_values_in_train`, `undefined_by_source`. `status=partial` означает, что
значение скрыто лимитом списка профиля (`outside_source_cap`).

`summary` перечисляет признаки с KS ≥ 0.1 и ≥ 0.3 (пороги `analyze_data.SHIFT_WARNING/CRITICAL`): вне 20 записей
профиля и с неизменной медианой. Инструмент ничего не удаляет и не преобразует.

Файлы: `shift_report.json`, `shift_features.csv`, `shift_summary.md`.

## Режим fit

Ядро — чистая функция:

```python
from experiment_round5_data_readiness import audit_oof_fit
audit = audit_oof_fit(X_fit, y_fit, context)
```

`context` такой же, как получает `execute_candidate`: `features`, `numeric`, `categorical`, `classes`,
`task_type`, `prediction_kind`, `metric`, `recipe`, а также `id_column` и `target_column` для проверки ролей.
Recipe должен пройти `checked_recipe` и быть `variant="oof"`. CLI строит действующий кандидат
`make_recipe(fixed_incumbent_recipe(), classes, variant="oof")`; на публичных классах его hash совпадает с
замороженным финалистом `cde4de87…`. Функция не открывает файлов и не видит TEST.

Порядок проверок:

1. **Контракт** — схема, порядок и роли признаков, ID/target не признаки, непустые строковые labels, все labels
   объявлены, порядок классов совпадает с recipe, классификация Macro F1 с метками. Нарушение — `ValueError`
   с причиной (отказ, а не результат).
2. **Grouped split** — `agent.round4_classification_candidate.coverage`: группы по всем raw-признакам, A/B через
   `agent.splits.split_rows` seed 42, три `StratifiedGroupKFold` в A. Если разбиение недоступно, причина берётся
   у того же `split_rows`: `coverage` сохраняет только тип ошибки.
3. **Пересчёт границ** — пересечения групп и строк A/B и fit/held каждого fold, строки вне A/B и вне A,
   покрытие A held-folds ровно один раз. Разбиению не доверяют: подменённое разбиение с утечкой ловится.
4. **Gates поддержки** — настоящая `fit_oof_guarded_rule` с `DEFAULT_GUARD_OPTIONS` из recipe. X — таблица без
   столбцов, callback отказывается обучать. Gates проходят до первого callback; их причина
   (`tuning_missing_class`, `selection_insufficient_class_support` и т. п.) переносится как есть.
5. **Размеры** — проверка входов адаптера вероятностей для каждого планового child-fit: три fold, selection
   (A → B) и fit-сторона финального fit. Причины: `row_cap_exceeded`, `feature_matrix_cap_exceeded`,
   `fit_missing_class`.

Решение `decision`:

- `structurally_compatible` — все проверки `passed`. Это значит только, что первый OOF-fit был бы начат.
- `ineligible` — есть хотя бы одно `violated`; `reasons` перечисляет `проверка:причина`.
- `unknown` — нарушений нет, но проверка закончилась неожиданно (например, отметки поддержки и gate разошлись).

`model_acceptance` всегда `not_evaluated`. В `not_evaluated` со статусом `unknown`: прирост Macro F1 на B,
class-drop guard, допуск по времени, успешный полный refit, caps predict-таблицы, память и равенство fit
будущему host INNER. Фиктивного `passed` у них нет.

Дубли raw-признаков считаются отдельно: групп с копиями, лишних копий, групп с разными labels и строк в них по
классам. Политика действующая: одинаковые признаки — одна группа при любых labels. Дубли не ошибка.

Файлы: `oof_fit_report.json`, `oof_class_support.csv` (часть × класс: строки и независимые группы для allowed,
A, B и fit/held каждого fold), `oof_fit_summary.md`.

## Что не экспортируется

Построчные ID, labels, индексы разбиений, ключи групп, примеры категорий и абсолютные пути. Разрешены имена
признаков, объявленные имена классов и агрегированные количества.

## Запись и хеши

- Папка вывода внутри `--data-dir` (shift) или внутри папок `task.md`, словаря и fit (fit) — отказ до записи.
- JSON-отчёт удаляется первым и пишется последним; CSV и Markdown пишутся атомарно (`tempfile` + `os.replace`).
  При ошибке записи на месте отчёта остаётся `status: error`, а набор считается недостоверным.
- Входы: shift — `dataset_fingerprint` DataCatalog; fit — SHA-256 и размер `task.md`, `data_dictionary.csv`,
  `fit.csv` под стабильными именами, с проверкой до и после чтения.
- Конфигурация: `config` и `config_sha256` (поля и пороги для shift; recipe hash, split, guards и caps для fit).
- Выход: `content_sha256` покрывает весь отчёт без времени и памяти; `files` — SHA-256 CSV и Markdown.
  Пути машины в хеши не входят, поэтому копия данных в другой папке даёт те же хеши.
- `generated` вне хеша: время UTC, `elapsed_seconds` и пик памяти процесса (`psutil` peak working set на Windows,
  `ru_maxrss` на Linux, иначе `unknown`).
