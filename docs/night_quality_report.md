# Ограниченный адаптивный поиск моделей

Ночная работа №1: production-политика `adaptive_v1`, доступная явной опцией. Confirmation показала выигрыш на MAE с тяжёлыми хвостами и меньшую стоимость, но ухудшение на signed MAE. Заранее заданный допуск общего default не пройден: **`legacy` остаётся default**, `adaptive_v1` используется явно. Все 72 сравнительных запуска завершены, включая полный Research v3 при отказе API и публичные train holdouts.

## Проблема и поведение

В прежнем поиске фиксированный список моделей делил время между множеством заранее предусмотренных попыток. Отдельный исследовательский портфель добавлял сильные варианты, но заметно увеличивал стоимость. Новая политика сначала проверяет разные семейства, затем ограниченно уточняет лучшие по inner validation. Для MAE существующие MAE objectives проверяются рано, обычные squared-loss модели остаются контролем.

После каждой попытки host измеряет wall time, включая подготовку, scoring и сохранение отчёта. Планировщик оценивает допустимость следующего trial по этим измерениям, размеру данных и остатку времени, оставляя резерв на outer и full refit. Это эвристическая оценка, а не гарантия стоимости модели; окончательные ограничения процессов и Research BudgetPolicy сохраняются.

Политика подключена к offline AutoML, стартовому AutoML в Research v3 и локальному продолжению после отказа API. LLM-кандидаты и прежние вызовы работают через существующие интерфейсы. Поиск ограничен 12 dispatch, initial diversity и максимум четырьмя предложенными refinement recipes; канонические дубли исключаются. Недоступные optional dependencies не запускаются повторно в AutoML.

Отдельные Macro F1 decision rules не включены. Preprocessing остаётся fold-local. Seed outer score не используется для inner ranking; отдельный inner fit того же рецепта всё ещё необходим. После заморозки финалистов выполняются terminal outer и штатный full-data refit. Recovery/checkpoints и ранний резервный CSV сохраняются.

## Версии и протокол

| Компонент | Версия |
|---|---|
| Baseline commit | `390829239bfb245e70d866c02d4cd69d51d4c0d9` |
| Измеренный candidate commit | `ac628cfcf4fc9d87573c226dbc393b52f4e66c73` |
| Baseline runtime SHA-256 | `423227637b7a0441e09d61954092dac8fb5069af5e4a4b2143f929af0d054b68` |
| Candidate runtime SHA-256 | `50b802838b615bd23aeb509e062de961874e6cc71af3fd3dfa627b8c603c7ceb` |

Последующие изменения этой ветки добавляют измеритель, тесты и результаты; runtime SHA совпадает с измеренной версией. Detached candidate и baseline остаются неизменными. Проверена одна версия политики; после confirmation новые параметры не подбирались.

До первого fit зафиксирован [протокол](night_quality_protocol.md). Development: 9 synthetic families × 2 seeds × 2 стратегии, budget 90 s, 36 runs. Confirmation: 6 заранее выбранных families × seeds 211/307 × 2 стратегии, budget 180 s, 24 runs. Baseline/candidate выполняются по очереди в свежих процессах; порядок чередуется между парами. Исходники обеих стратегий, host harness, библиотеки, inputs/splits/audit hashes заморожены до запуска. Модельные параметры не меняются после просмотра confirmation.

Дополнительно заранее подготовлены 4 полных Research v3 run с имитируемым отказом API и 8 run на holdout частях публичного train: grouped multiclass и past-only regression. Публичные train уже использовались в предыдущих исследованиях; они служат sanity-проверкой, а не независимым confirmation или платформенным score. Всего запланировано 72 измерительных run, сумма process wall caps 9720 s; общий резерв стендов ограничен. Отдельный flat-root CLI integration выполняется во временном ZIP.

Никакие audit answers не передаются solver-коду или подбору рецептов. Обычные процессы здесь не являются OS sandbox. Outer audit хранятся вне task directories. На этом host недоступна достоверная process memory/CPU telemetry: представлены wall times, а не измеренные пиковые RAM/CPU. Нативные resource checks отдельно прошли в CI.

## Результаты

### Development, budget 90 s

Все 36 run завершены, все CSV корректны; 18 пар: 8 wins / 6 ties / 4 losses. Общий solver wall time: baseline 387.957 s, candidate 339.600 s (меньше на 12.465%). Метрики в таблице — медианы двух seeds, gain — медиана **парных** относительных снижений loss либо абсолютных изменений F1. W/T/L относится к отдельным seeds.

| Family | Метрика | Baseline | Candidate | Медианный gain | W/T/L |
|---|---|---:|---:|---:|---|
| linear_regression | RMSE | 0.910712 | 0.910712 | 0% | 0/2/0 |
| nonlinear_regression | MAE | 5.413730 | 2.356291 | +56.61% | 2/0/0 |
| temporal_drift | RMSE | 0.493957 | 0.495981 | −0.47% | 0/1/1 |
| imbalanced_binary | LogLoss | 0.335575 | 0.331232 | +1.54% | 2/0/0 |
| multiclass_duplicates | Macro F1 | 0.911328 | 0.911328 | +0.000000 | 0/2/0 |
| categorical_shift | LogLoss | 0.365394 | 0.371426 | −1.63% | 1/0/1 |
| mae_signed_nonlinear | MAE | 0.423586 | 0.418372 | +1.22% | 1/0/1 |
| mae_symmetric_linear | MAE | 0.661316 | 0.663471 | −0.32% | 0/1/1 |
| mae_heavy_tailed | MAE | 1.720823 | 1.477123 | +14.13% | 2/0/0 |

В nonlinear/heavy-tailed победили MAE objectives или blend MAE моделей. Это согласуется с исходной гипотезой, но здесь измерен совместный эффект порядка, портфеля и scheduler; отдельной objective-only ablation не было. На categorical seed 73 LogLoss ухудшился с 0.368719 до 0.381280; этот seed сохраняется в сравнении.

Исходный development report SHA-256: `4f6739de221a6df56d59ca52037cb4cc1802dd216d94a2e9af3d6a74d957f579`. V1 full-refit checker подтвердил 18/18 baseline и 17/18 candidate: trial 1 не прошёл только требование текстового stdout. Его CSV и NPZ побайтно совпадают с baseline, но исходный flag не меняется. Отдельный v2 re-audit подтвердил 36/36: проверены исходники, launcher/config, полные inputs, end-of-worker durations, accepted checkpoint и повторно вычислена метрика. Причина пустого stdout не установлена. Новый audit SHA-256: `2379aebc41faf126f8fe967673203108beca632dabd5a1c21f799604565f88d6`; исходный report неизменен.

### Confirmation, budget 180 s

Все 24 run завершены штатно, CSV и full refit подтверждены v1/v2. 12 пар: 6 wins / 2 ties / 4 losses. Solver wall time: baseline 260.791 s, candidate 225.051 s (меньше на 13.7045%). Два новых seeds на family, правила агрегации те же.

| Family | Метрика | Baseline | Candidate | Медианный gain | W/T/L |
|---|---|---:|---:|---:|---|
| linear_regression | RMSE | 0.748610 | 0.747977 | +0.08% | 2/0/0 |
| mae_signed_nonlinear | MAE | 0.459847 | 0.481831 | **−4.77%** | 0/0/2 |
| mae_heavy_tailed | MAE | 1.778149 | 1.580420 | **+11.11%** | 2/0/0 |
| imbalanced_binary | LogLoss | 0.286211 | 0.287067 | −0.21% | 1/0/1 |
| multiclass_duplicates | Macro F1 | 0.910390 | 0.910390 | +0.000000 | 0/2/0 |
| categorical_shift | LogLoss | 0.416997 | 0.427893 | −2.65% | 1/0/1 |

Signed MAE проиграла на обоих seeds: −4.1780% и −5.3687%, медиана хуже допуска −3%. Практически значимый медианный выигрыш ≥5% есть только у heavy-tailed family; требовались две. Categorical seed 307 отдельно ухудшился на 7.0372%; этот результат также сохранён, хотя медиана family укладывается в заданный допуск. Поэтому общий default не меняется. Выигрыш на synthetic heavy-tailed MAE не доказывает рост private score.

Исходный confirmation report SHA-256: `96b32e1cb23332f3a6bf1a5cdabd33fcc493e9037818004982ba58bb7c78f02e`. После просмотра confirmation runtime не менялся.

### Сквозные проверки

Research v3: по одному новому seed 811 для regression и binary, budget 180 s. Все четыре run завершились с корректным CSV, terminal outer, full refit и действительными receipts; повторная v2 проверка согласовалась с исходным score. Две имитируемые ошибки соединения на run (первая попытка и один retry), реальные внешние API requests: 0.

| Задача | Метрика | Baseline | Candidate | Loss gain | Wall time B/C, s |
|---|---|---:|---:|---:|---:|
| nonlinear_regression | MAE | 5.412838 | 4.452381 | +17.74% | 29.944 / 29.276 |
| imbalanced_binary | LogLoss | 0.385631 | 0.324970 | +15.73% | 32.462 / 34.063 |

Реальный локальный continuation candidate выполнил 6 и 5 verified inner experiments соответственно; в binary один из них улучшил inner result. Эти две integration-пары проверяют полный host path и не заменяют независимую confirmation-серию.

Public sanity: исходные публичные test labels неизвестны. На regression используются два past-only holdout периода, на classification — два grouped outer split публичного train. Public train уже изучался раньше; эти четыре пары не являются новым independent confirmation. RMSE и F1 не усредняются между собой.

| Public TRAIN holdout | Метрика | Baseline | Candidate | Gain | Wall time B/C, s |
|---|---|---:|---:|---:|---:|
| regression, period 4 | RMSE | 44.761238 | 55.244543 | **−23.42%** | 40.124 / 4.818 |
| regression, period 5 | RMSE | 48.994980 | 43.980237 | +10.24% | 71.982 / 52.405 |
| classification, seed 1729 / fold 0 | Macro F1 | 0.619064 | 0.631039 | +0.011975 | 136.605 / 138.151 |
| classification, seed 2718 / fold 0 | Macro F1 | 0.619291 | 0.642213 | +0.022922 | 136.264 / 113.930 |

Все 8 run завершены с корректным CSV и полным refit, 4 пары: 3 wins / 0 ties / 1 loss. Для regression period 4 подтверждён важный недостаток admission heuristic: inner fit 2117 rows, outer fit 11175, full fit 22054. Неизвестная стоимость 5 s умножается на отношения размеров в степени 1.25; резерв outer+refit получается **202.378 s при остатке 175.403 s**. Все 12 предложений отклонены, остаётся проверенный seed Ridge. Он штатно обучен на всём доступном train, CSV корректен, wall cap не превышен, но RMSE хуже на 23.42%. `search_status=time_limit` здесь отражает остановку admission, а не настоящий solver timeout. Снижение общей стоимости public нельзя считать самостоятельным выигрышем: часть экономии получена именно отказом от поиска на этом случае.

Уже завершённый seed имел host cost 3.216 s и worker durations fit 1.4592 / predict 0.4993 / total 2.0723 s. Scheduler не использовал это timing evidence, поскольку его текущие observations относятся к inner trials. Для следующей версии нужны отдельные source/config/rows-bound timing priors, включая seed, и разделение постоянных process overhead и стоимости fit/predict. При этом отдельный inner fit и запрет seed outer score в ranking должны сохраниться. Эти изменения **не внедрялись** после просмотра confirmation/public results.

Полные значения всех 36 пар, inputs/audit/source hashes, library versions, wall times, v1/v2 refit counts и проверка численного допуска находятся в [night_quality_results.json](night_quality_results.json). Все **72/72 solver run** завершены штатно с корректным CSV, неизменными inputs/source и подтверждённым full refit по отдельному v2 re-audit. Solver timeouts/failures и реальные API requests: 0. Суммарный solver wall time 2033.424 s; это включает preprocessing, subprocess overhead, scoring и финализацию, не является чистым временем fit.

Исходные report hashes: Research `ca44a0b8de50359701bb8a46dde9fede830849df71b42005854227515b1e791f`, public `1b053eebe309437a6c33dade56517708fd6837949ac7e8acc42f4e589e1b1407`. Исторические v1 flags не переписаны. Новый full-refit auditor SHA-256 `b5f95610d69cfce9217fd65230c40441635951d7aa213e05371ebe21c4cc8ec2`, descriptive summary auditor `4138bc82e8be15ea3c681d6807d90816dfe49ca87f1d3beb82d4a15162270548`. Re-audit не обучает новые модели и не меняет старые файлы.

## Проверки совместимости

На исходном candidate `dev_check.py`: 645 tests, 412.425 s, OK (5 локальных skips). PR #14 CI run `39525138920`:

| Среда | Core | App | Launcher и ZIP |
|---|---|---|---|
| Ubuntu | 645, 525.324 s, 2 skips | 227, 76.258 s, 3 skips | success |
| Windows | 645, 647.412 s, без skips | 227, 115.567 s, 6 skips | success |

Все 65 исходных policy/schedule/integration/local/runner tests прошли на обеих ОС. После добавления public runner, строгого re-audit, агрегатора и настоящего flat CLI test финальный локальный `dev_check.py`: **667 tests, 436.295 s, OK (5 skips)**. В него входят все 87 профильных night-quality tests, включая реальный flat ZIP: исходные ID, ранний median prior, отличные от prior обученные predictions, full-data worker/config/durations/checkpoint и независимый score. Пропуски относятся к OS-specific/resource telemetry checks локального host; отсутствие telemetry не считается измеренным RSS=0.

Штатный workflow повторяет core/app/launcher/ZIP проверки для финального head на Linux и Windows. Текущие результаты и логи: [PR #14 checks](https://github.com/NolReaction/alfa-automl-agent/pull/14/checks). Исходный `delivery/solution.zip` во время этих локальных проверок не перезаписывался.

## Включение

Ограничения текущей версии: неизвестное семейство начинает с оценки 5 s и первой попытки максимум около 10 s даже при большом общем budget. На больших данных это может исключить полезную модель до получения успешного timing. Оценки по семейству, глубине и числу строк приближённые. Локальное Research-продолжение начинает новые измерения стоимости, используя startup только как порядок предпочтений; неудачный optional import там может занять несколько ограниченных попыток. AutoML исключает такое семейство после первого отказа. Трёхчасовые поиски в этой серии не измерялись.

```powershell
.\.venv\Scripts\python.exe solution.py data\regression\task.md --mode offline --time-budget 180 --search-policy adaptive_v1
```

Та же опция поддерживается в `--mode llm`. Без неё действует `legacy`; интерфейс платформы `python3 solution.py task.md` сохраняется. Новый default не допускается по результату независимой confirmation.

## Воспроизведение

Профильные проверки (включают один реальный запуск из временного flat ZIP):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_night_quality_*.py" -v
```

Для нового парного измерения создайте чистые detached worktrees от указанных SHA. Harness запускается из чистой текущей ветки; output directory должна быть новой. Подготовка не обучает модели. Каждый solver run получает тот же budget; replay выполненного plan запрещён.

```powershell
git worktree add --detach ../night-baseline 390829239bfb245e70d866c02d4cd69d51d4c0d9
git worktree add --detach ../night-candidate ac628cfcf4fc9d87573c226dbc393b52f4e66c73
.\.venv\Scripts\python.exe experiment_night_quality.py --prepare-only --output data/experiments/night_quality/repro-confirm --baseline-root ../night-baseline --candidate-root ../night-candidate --cases linear_regression mae_signed_nonlinear mae_heavy_tailed imbalanced_binary multiclass_duplicates categorical_shift --seeds 211 307 --budgets 180 --purpose confirmation --max-total-seconds 4440
.\.venv\Scripts\python.exe experiment_night_quality.py --run-prepared data/experiments/night_quality/repro-confirm
.\.venv\Scripts\python.exe experiment_night_quality.py --reaudit-prepared data/experiments/night_quality/repro-confirm --reaudit-output data/experiments/night_quality/repro-confirm-reaudit.json
.\.venv\Scripts\python.exe summarize_quality.py --report data/experiments/night_quality/repro-confirm/report.json --reaudit data/experiments/night_quality/repro-confirm-reaudit.json --json-output data/experiments/night_quality/repro-confirm-summary.json --markdown-output data/experiments/night_quality/repro-confirm-summary.md
```

Для development используйте все 9 families из протокола, seeds 42/73, budgets 90, purpose development, max-total-seconds 3360. Для integration — cases nonlinear_regression imbalanced_binary, seed 811, budget 180, modes research_api_failure, purpose integration, max-total-seconds 780. `experiment_night_public.py --help` описывает public preparation: classification seeds 1729/2718, regression periods 4/5, budget 180, memory-mib 8192, max-total-seconds 1560. Подготовьте официальные данные штатным `prepare_data.py` и передайте оба task.md. Public run не использует исходные test labels.

Исходный отправленный `delivery/solution.zip` не изменялся. Main не объединяется автоматически; результат готовится отдельным PR #14 для совместного review с ночной работой №2. Реальных запросов к конкурсному API и отправок на платформу: 0.
