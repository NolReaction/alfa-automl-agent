# XGBoost-кандидат v1: контракт, проверки и native smoke

Дата: 22 октября 2026, МСК. Исполнитель: Мурад. ТЗ: `docs/team_round4/MURAD_XGBOOST.md`.

Итог: изолированный CPU-кандидат `xgboost_native_v1` для регрессии, бинарной и
многоклассовой классификации готов к общей проверке. Он не подключён к поиску,
не включён по умолчанию и не меняет существующие модули. Прирост качества не
измерялся и не заявляется: smoke ниже проверяет механику протокола на
синтетике, написанной автором кандидата.

## База, ветка и hashes

- Точная база: `184408390c83ef951e5a570cf9d2d6c3e9e72f98` (ветка PR #27 `work/finalist-selection-v1`).
- Ветка: `work/murad-xgboost-v1`, только 6 новых файлов из ТЗ. Существующие файлы не изменены.
- SHA-256 файлов, которые использовал smoke (совпадают с `plan.json` и с коммитом ветки):

| Файл | SHA-256 |
| --- | --- |
| `agent/xgboost_candidate.py` | `531c471b73c082e9681fd4f9695978b8a2f9f976e123ea5f37cfe45816e7c69a` |
| `xgboost_candidate_cases.py` | `099bf7c58fe4d9e51b3c3f7681ff79ade52e2408f4bb16eedd5cda6a322465d4` |
| `experiment_xgboost_candidate.py` | `f8775c8dd5f57fdb5be513b8ee71d0dac77f8be99691083fc41b6be3c5000fcb` |
| `tests/test_xgboost_candidate.py` | `54f7141c8f357426b00b2089ca2513e85a3a0532a6d66afbf0cc9d48edfbbd02` |
| `tests/test_xgboost_candidate_experiment.py` | `89848cbc4618ace21166828080c0e9dcf940b9d8e126d70812cc1bdb9ece54f7` |

`plan.json` smoke: `819dea1a036d2f98377e719d58fc01bb3d057ef717c0fc75dc6751f4068c49a8`,
`results.json`: `ea160b222d425243a4c9460ab2bc603a5885623d108aa0b2776078703b1de635`.
В плане записан head `1844083`: smoke шёл до коммита, поэтому файлы
сверяются по hashes, а не по SHA коммита.

## Окружение

Windows 11 Pro, 8 логических CPU, Python 3.12.0. Пакеты: numpy 1.26.4,
scipy 1.12.0, pandas 2.2.3, scikit-learn 1.5.2, joblib 1.4.2,
threadpoolctl 3.6.0, psutil 6.1.1, xgboost 2.1.4. Версия xgboost совпадает с
`requirements.platform.txt`. Её я поставил отдельно в существующий `.venv`
командой `pip install --only-binary=:all: xgboost==2.1.4`, других пакетов
это не добавило. В wheel есть сборка с CUDA; кандидат всегда задаёт
`device="cpu"`.

API 2.1.4 проверен по установленному пакету: сигнатуры `xgboost.train`,
`DMatrix`, `Booster.predict`, `callback.EarlyStopping`. Проверена и официальная
документация: `xgb.train` с early stopping возвращает последнюю модель, а не
лучшую; `best_iteration` считается с нуля. Поэтому кандидат берёт
`best_iteration + 1` деревьев и обучает новую модель.

CI ставит `requirements.txt`, где xgboost нет. Там 12 native-тестов
пропускаются через `skipUnless`, остальные 27 проверок контракта выполняются.
Модуль импортирует xgboost только при обучении. Все проверки допуска (схема,
caps, split, классы) выполняются и без xgboost.

## Контракт

```python
recipe = make_xgboost_recipe(task_type, metric_name, validation=policy,
                             parameters=FIXED_CONFIGURATIONS["depth6_v1"])
source = candidate_source()
```

Recipe — конечный JSON: `version=1`, `family="xgboost_native_v1"`, задача,
метрика, loss, полный набор параметров, `feature_config`, `preprocessing`,
политика split, `max_iterations`, `stopping_rounds` и `runtime` (`hist`, `cpu`,
до 8 потоков, seed 42, версия 2.1.4). При исполнении recipe собирается
заново и должен совпасть с нормализованным. Любое лишнее поле или подмена
`device`/`loss`/`family` отвергается до обучения. Входы и context не мутируются.

- Whitelist параметров: `learning_rate` 0.01–0.3, `max_depth` 1–10,
  `min_child_weight` 0–100, `subsample` 0.5–1, `colsample_bytree` 0.5–1,
  `reg_lambda`/`reg_alpha` 0–100, `gamma` 0–10, `max_bin` 16–256. Bool, строки,
  NaN/inf и float вместо int отвергаются. Objective, eval metric, потоки,
  seed, устройство и callbacks из JSON не принимаются.
- Две фиксированные конфигурации: `depth6_v1` (по умолчанию) и
  `depth4_regularized_v1`. HPO нет.
- Политика split обязательна: `random`, `time` (только регрессия) или
  `grouped` (только классификация), как у boosting-кандидата. `None`/`auto`
  не подменяются случайным split.
- Caps проверяются до preprocessing и обучения: строки fit/predict (1 млн),
  выходные признаки (2048), ячейки матрицы (50 млн), уровни категорий
  (500 тыс.), классы (50), деревья `max_iterations × классы` (40 тыс.).
  Выход за cap даёт `skip resource_cap_exceeded:<cap>`, данные не урезаются.
  Это пороги допуска, а не измеренные пики и не sandbox.

Один вызов `fit_predict` (одинаковый для INNER, OUTER и full refit; phase
только пишется в evidence):

1. Проверка recipe, метрики, схемы признаков и исходного deadline хоста.
   Пропуск объявленного класса в fit даёт `skip fit_missing_class`. Метка вне
   объявленных классов — ошибка контракта.
2. Внутренний split через существующий `split_rows`, только по X_fit/y_fit.
   Для времени held — последний период, fit — строго прошлые периоды. Для групп
   дубли признаков не пересекаются. Невозможная политика даёт
   `skip internal_split_unavailable`, класс, пропавший в одной из частей, —
   `skip internal_split_missing_class`.
3. Preprocessing обучается на внутренней fit-части: `FeatureTransformer`, затем
   `models._encoded_preparation(..., linear=False)`.
4. Early stopping на held, затем новый preprocessor и новый booster на всех
   X_fit/y_fit ровно с `best_iteration + 1` деревьями. Число деревьев не
   ограничивается снизу.
5. Вероятности переставляются в literal-порядок классов хоста. Binary
   превращается в два столбца, float32 перенормируется после проверки
   отклонения ≤ 1e-4. Для метрик-меток берётся argmax этих же вероятностей.

Оценка `_encoded_preparation(linear=False)`: подходит для v1. Медианы и словари
OrdinalEncoder учатся только на fit-строках. Строки `"NA"`, `"0007"`, `"7"`
различаются, пустое значение — отдельная категория. Невиданные категории
получают `-1`. One-hot нет. Цена:

- медианное заполнение стирает признак «пропуск» у числовых столбцов, и
  native NaN-обработка XGBoost не используется. `missing_indicators=true` в
  `feature_config` возвращает этот сигнал явно.
- ordinal-коды упорядочены лексикографически, это не native categorical.

Native categorical и sparse semantics в v1 сознательно не включены.

Критерий остановки и его связь с метрикой хоста:

| Метрика хоста | Objective / stopping | Связь |
| --- | --- | --- |
| RMSE | `reg:squarederror` / `rmse` | та же величина |
| MSE, R2 | `reg:squarederror` / `rmse` | монотонно эквивалентна на фиксированном held |
| MAE | `reg:absoluteerror` / `mae` | та же величина |
| LogLoss | `binary:logistic`/`multi:softprob` / `logloss`/`mlogloss` | та же величина (с точностью до клиппинга) |
| Macro F1, Weighted F1, Accuracy, ROC AUC | те же logloss/mlogloss | **proxy**, не метрика хоста |

Время. Внутреннее обучение получает не больше 30% остатка; доля уменьшается
при дорогом refit (та же эвристика, что у boosting-кандидата). Перед refit
оценка его стоимости сравнивается с остатком исходного deadline. Если времени
мало, кандидат завершается ошибкой `deadline_exhausted` на стадии
`refit_admission`. Если deadline наступает во время refit, это ошибка
`refit_training`, а не модель с меньшим числом деревьев. Callback проверяет
время между итерациями. Одну итерацию он не прерывает, окончательный
ограничитель — supervisor хоста (≤ 1800 с на worker).

Evidence (`xgboost_evidence.json`, режим `"x"`, один файл на вызов):

- версия пакета и frozen recipe;
- split policy и размеры;
- число деревьев: обучено внутри, best, selected, обучено в refit;
- `stopped_by`, objective, stopping metric и её связь с метрикой хоста;
- строки full-fit и predict, hash выученных медиан и словарей (внутренний и refit);
- время стадий, оценка refit (`estimate_guaranteed=false`), остаток deadline;
- `nthread`, caps (`configured_limits_are_measured_peaks=false`).

При skip или нехватке времени пишется `skipped`/`failed` с причиной, стадией и
частичным trace, после чего ошибка пробрасывается хосту. Evidence — диагностика,
а не оценка качества: метрику хост считает сам по своим ответам.

## Проверки

Локально с xgboost 2.1.4 прошли 39 новых тестов: 27 в
`tests/test_xgboost_candidate.py` и 12 в `tests/test_xgboost_candidate_experiment.py`.
С заблокированным xgboost (как в CI) тоже 39 OK: 12 native пропущены. Покрыто:

- регрессия (time), бинарная и многоклассовая классификация; MAE со второй
  конфигурацией; пропуски и новые predict-категории;
- порядок вероятностных столбцов для прямого и обратного порядка классов
  (2 и 3 класса); метки = argmax тех же вероятностей;
- отсутствие held-класса, пропуск fit-класса, неизвестная метка, неизвестные и
  ресурсные параметры, bool вместо числа, подменённый recipe;
- внутренний preprocessing видел только внутренние fit-строки: held-only и
  predict-only категории не в словаре; финальный refit видел все fit-строки;
- time: max периода внутреннего fit < min периода held; grouped: hashes
  групп не пересекаются;
- другой predict-batch и другая phase не меняют hashes preprocessing, выбор
  деревьев и best score;
- истёкший deadline до любого fit; остановка перед refit по оценке; deadline
  внутри refit; ресурсные skip без обучения; неизменные входы; нет файлового
  I/O при прямом вызове;
- настоящий subprocess через `run_candidate`: точные столбцы CSV, форма
  вероятностей, receipt, независимый пересчёт метрики по сохранённому CSV,
  full refit; native skip пишет receipt и не создаёт predictions;
- новые runtime-файлы не импортируют сетевые/API-модули и не читают `.env`/environ.

### Полный набор тестов

`python -m unittest discover -s tests` на ноутбуке Мурада с установленным
xgboost: 1260 тестов за 2173 с, 2 failures, 11 errors, 7 skipped. Новых тестов
среди упавших нет. Новые файлы не импортируются существующими модулями.
Разбор падений:

- 8 errors в `test_execution_paths` и 1 в `test_research_deep_paths`:
  `FileNotFoundError [WinError 3]` на глубоких путях. На машине
  `LongPathsEnabled=0`, это ограничение длины пути Windows.
- `test_research` (API-failure case), `test_offline_archive`: падают и при
  повторе с заблокированным импортом xgboost. Эти же тесты падали на чистой
  базе на этом ноутбуке и в прошлых раундах.
- `test_frequency_experiment` (subprocess timeout 20 с) и
  `test_research_v3_archive` (`seed_only` вместо `searched`): при отдельном
  повторе с заблокированным xgboost оба прошли. Это зависимость от времени
  при полной загрузке машины.

Окончательная проверка — CI ветки на Windows runner без xgboost.

## Native smoke

Каталог и конфигурации записаны до запуска командой `prepare`: 4 случая,
recipes, входные CSV, hashes sources и окружения. Ответы holdout лежат в
`host_only/`, test-ответов нет. `run` прогоняет случаи последовательно, каждый
в новом каталоге, через v3 `run_candidate`. Сначала `outer_validation` (fit
хоста → holdout, метрику считает хост), затем `full_refit` (fit + holdout →
test, без ответов). Ошибки и timeouts сохраняются, повторов нет. `audit`
заново хеширует outputs и receipts и пересчитывает метрику без обучения.

Команды (PowerShell, из `..\alfa-xgboost`; в worktree нет своего `.venv`):

```powershell
..\alfa-automl-agent\.venv\Scripts\python.exe -m unittest tests.test_xgboost_candidate tests.test_xgboost_candidate_experiment -v
..\alfa-automl-agent\.venv\Scripts\python.exe experiment_xgboost_candidate.py prepare --output data\experiments\xgboost_candidate\smoke_20261022_v1
..\alfa-automl-agent\.venv\Scripts\python.exe experiment_xgboost_candidate.py run --output data\experiments\xgboost_candidate\smoke_20261022_v1
..\alfa-automl-agent\.venv\Scripts\python.exe experiment_xgboost_candidate.py audit --output data\experiments\xgboost_candidate\smoke_20261022_v1
```

Повтор у Димы: те же три команды с `python` своего окружения и новым
каталогом под `data\experiments\xgboost_candidate\`. Существующий каталог
`prepare` и `run` не перезаписывают.

Запуск 2026-10-21 21:50:00–21:50:43 UTC (00:50 МСК): 8 из 8 фаз
`completed`, все проверки true, `audit`: `all_matched=true`, `trained=false`.

| Случай | Метрика, конфигурация | Внутренний split | OUTER: метрика / константа | Деревья OUTER (обучено/выбрано/refit) | Деревья full refit | Остановка | Время child, с (OUTER / refit) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `regression_time_rmse` | RMSE, depth6 | time: ≤4 → 5; refit ≤5 → 6 | 0.8875 / 2.6773 | 370 / 320 / 320 | 239 / 189 / 189 | early stop, rmse (та же) | 4.81 / 4.55 |
| `binary_random_auc` | ROC AUC, depth6 | random 1536/384 | 0.8807 / 0.5 | 143 / 93 / 93 | 127 / 77 / 77 | early stop, logloss (proxy) | 4.31 / 4.34 |
| `multiclass_grouped_macro_f1` | Macro F1, depth4_reg | grouped 768/192 групп, overlap 0 | 0.8426 / 0.1472 | 431 / 381 / 381 | 265 / 215 / 215 | early stop, mlogloss (proxy) | 5.51 / 5.06 |
| `regression_random_mae` | MAE, depth4_reg | random 1536/384 | 0.8131 / 1.9814 | 369 / 319 / 319 | 252 / 202 / 202 | early stop, mae (та же) | 4.59 / 4.51 |

Строки OUTER: fit 2000 / holdout 400 для time-случая, 1920 / 480 для остальных.
Full refit: 2400 / test 400. «Константа» — эталон из fit-ответов (среднее,
медиана, priors или самый частый класс); это проверка, что модель чему-то
научилась, а не соперник. Сам кандидат внутри child занимал 0.53–1.91 с,
остальное время — запуск процесса и импорты. Пиковый RSS по выборкам psutil
хоста — 256–267 MiB. Это измерение по 38–48 выборкам, а не гарантированный
пик и не жёсткий лимит (настроенный cap хоста 28 GiB). Receipts хоста и
`xgboost_evidence.json` сохранены у каждой фазы. Пересчёт хоста по
сохранённому CSV совпал с метрикой `run_candidate` во всех 4 OUTER-фазах.

Проверки утечек в smoke:

- `fit.csv` OUTER-вызова не содержит ID holdout, а в `predict.csv` нет столбца цели;
- internal time boundary строго прошлая;
- grouped `group_overlap=0`;
- `full_fit_rows` и `predict_rows` в evidence совпадают с тем, что передал хост;
- в refit обучено ровно выбранное число деревьев;
- файлов с ответами внутри `runs/` нет.

## Ограничения

- Синтетические случаи написал автор кандидата. Это проверка механики, а не
  независимый quality audit. Сравнения с ExtraTrees, AutoGluon или полным
  агентом не было, превосходство не заявляется.
- Для F1/Accuracy/ROC AUC остановка идёт по logloss-proxy. Выбранное число
  деревьев оптимально по proxy, а не по метрике хоста.
- `subsample`/`colsample_bytree` < 1 используют RNG с фиксированным seed.
  Повторяемость между машинами с разным числом потоков отдельно не проверялась.
- Медианное заполнение и ordinal-коды (см. выше) — осознанное упрощение v1.
- Оценка времени refit — эвристика без гарантии. Одна итерация XGBoost не
  прерывается посередине.
- Классификация с time-политикой и регрессия с grouped не поддерживаются, как
  у boosting-кандидата.
- Unit tests, `--help` и этот smoke не являются ML-экспериментом качества.
  Замер пользы — у лаборатории и общей приёмки.

Ноль API: модуль и эксперимент ничего не скачивают, не обращаются к API и
внешним моделям, не читают `.env`; платформенных отправок нет. Сырые данные,
модели и логи лежат в игнорируемом `data/experiments/xgboost_candidate/`.

## Для подключения (Дима и основной ИИ)

Подключение к общему поиску в этом PR не делалось. Естественный вход — аналог
`_boosting_trial` в `agent/research_extensions.py`:
`make_xgboost_recipe(spec.metric.task_type, spec.metric.name, validation=gate.choice, parameters=FIXED_CONFIGURATIONS[...])`
и `candidate_source()`. Решение о defaults, бюджете и приёмке остаётся у вас.
