# ИИ 2: проверка архитектуры на одинаковом CPU-бюджете

Нужен следующий отдельный PR: лабораторный эксперимент, который покажет,
помогает ли сложность нынешнего агента по сравнению с простыми сильными моделями
и прямым native AutoML. Сначала проверяем методику, затем получаем реальные
измерения там, где доступно достоверное наблюдение процессов и памяти.

## База, ветка и границы

- Начни от `216f17ab0c00486c0f3f9a7676e682c43b030c16`.
- Ветка: `work/architecture-ablation-lab-v1`.
- Твои файлы: новые `research_lab/ablation/**`,
  `tests/test_research_lab_ablation_*.py`, `docs/research_lab_ablation.md`.
- Весь существующий v1, его файлы, планы и результаты остаются замороженными.
- Production, `agent/**`, `solution.py`, requirements, CI и сборщик ZIP не менять.
- Не реализуй новую ML-библиотеку и не исправляй архитектуру одновременно с её измерением.
- Не запускай Qwen/API, GPU, аренду, платные ресурсы или отправку на платформу.
- Не делай merge самостоятельно; принеси готовый PR с проверяемыми результатами.

Отдельная точка входа: `python -m research_lab.ablation ...` либо равноценная
документированная команда внутри нового пакета. Не расширяй старый CLI ценой
изменения семантики существующих sealed-планов.

## Что именно сравниваем

Во всех четырёх arm ML-код берётся из одного чистого checkout:
`a45861d7e5588288db66f9ae088a9726f9597370`.
Никаких finalist-selection, strong search, новых feature-рецептов или PR #27.

| Arm | Метод |
| --- | --- |
| `fixed_catboost` | Одна заранее заданная CPU-модель CatBoost с TRAIN-only early stopping и свежим полным refit. |
| `fixed_lightgbm` | Одна заранее заданная CPU-модель LightGBM с тем же правилом границы данных. |
| `direct_automl` | Прямой существующий `run_automl` через native entry point. |
| `full_agent_local` | Текущий полный CPU-путь `run_agent(local_only=True)` с adaptive local search. |

Native аргументы для `direct_automl`:
`--mode offline --research-search legacy --search-profile legacy`.
Для `full_agent_local`:
`--mode offline --research-search adaptive --search-profile legacy`.
В обоих случаях вызывается настоящий `solution.py`, которому передаётся только
остаток исходного дедлайна через `--time-budget`.

Назови arm явно `full_agent_local`: это не измерение вклада Qwen или задержек API.
Не заставляй все методы иметь одинаковое внутреннее число split/refit:
их фактическая стоимость и результат являются предметом сравнения.
Внешний audit-набор, разрешённые входы и общий бюджет одинаковы.

## Фиксированные модели: полностью определить до запуска

Используй существующие `make_model_config`, `normalize_feature_config`,
`agent.boosting_candidate.make_boosting_recipe` / `execute_boosting_candidate`.
Исполнение — только в свежем child из указанного solver checkout.
Trusted аудитор не импортирует Python из переданного solver-root.

Зафиксируй следующие стартовые параметры без подбора по audit:

| Семейство | Параметры |
| --- | --- |
| CatBoost | `iterations=1200`, `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=5.0`; без auto class weights. |
| LightGBM | `n_estimators=800`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=3.0`, `max_depth=-1`, `colsample_bytree=1.0`; без class weights. |

Для early stopping максимальное число итераций — соответственно 1200/800,
`stopping_rounds=50`; остальные правила существующего boosting executor сохраняются.
Feature config — нормализованный обычный config без дополнительных преобразований.
Model seed — существующий native `42`, dataset/split seed — `1729`.
Полный нормализованный config, objective, metric и версии попадают в method manifest.
Никаких автоматически подобранных class weights, thresholds, blends или target transforms.

Внутренний split строится только из текущего TRAIN по существующему host chooser:
зафиксируй алгоритм выбора random/group/time и сохрани получившийся config до fit.
`X_predict` и внешние audit-ответы не выбирают число деревьев или параметры.
Imputation, категории и остальные обучаемые преобразования fit только на fit-части.
После early stopping нужен новый pipeline, обученный на всём предоставленном TRAIN,
затем predict test. Сохрани внутренние размеры, выбранные итерации и полный refit.
Не сохраняй модель, обученную только на early-stopping fit-части, как конечный результат.
Не выбирай «лучший из двух fixed» по внешнему audit: это два самостоятельных arm.
Не расширяй fixed-arm новыми пробами, чтобы искусственно израсходовать весь бюджет.

Если existing executor неприменим, запиши явный skip/failed с причиной;
не подменяй семейство, split, seed или бюджет после просмотра результата.
Не используй monkeypatch native search как способ получить «одну модель».

## Данные и внешний audit

Переиспользуй существующие verified cache, catalog и adapters без изменений.
Четыре development-набора: `bank_marketing`, `wdbc`, `wine_quality_red`, `auto_mpg`.
Один seed `1729`; одинаковые байты TRAIN/test/TaskSpec во всех четырёх arm.
Сохрани existing правила исключения leakage, группировки дублей и split.

Это знакомые development-данные. Bank diagnostic LogLoss уже раскрывался;
данные/метрики нельзя называть новой независимой confirmation-выборкой.
Не запускай confirmation, suite или 10 500-секундный этап в рамках задания.
Не включай исходный contest public TRAIN в этот четырёхдатасетный результат.

Audit labels находятся вне solver task directories, argv, config, env и task text.
До fit заморозь роли данных, split/inputs/source hashes, arm configs, порядок,
метрики, правила сравнения, ресурсы, solver/auditor commits и environment.
Новая версия метода требует нового чистого commit и нового плана.
Старый censored pilot не переписывать, не возобновлять под другим методом.

## Ресурсы, дедлайн и журнал

На arm: максимум 360 секунд общего wall time, 8 CPU threads,
configured child-tree RSS cap 24 GiB на подходящем ПК с 32 GiB RAM.
16 arm × 360 s = **96 минут training cap**, плюс отдельно preparation/cleanup.
До запуска покажи estimate; не начинай больший этап автоматически.

Один исходный дедлайн начинается до Popen/импортов. Он включает startup,
emergency prior, preprocessing, search, внутренние проверки, полный refit,
predict, публикацию и обязательную запись evidence. Cleanup учитывается отдельно.
Для adaptive native invocation после импортов должно остаться >300 секунд.
Не выдавай nested-процессам новый независимый training budget.
Быстро завершившийся метод не обязан ждать истечения 360 секунд.

Все ML-arm выполняются последовательно; не запускай тяжёлые тесты/fit параллельно.
Порядок четырёх arm зафиксируй до результатов, например циклический сдвиг
по четырём datasets, чтобы каждый arm занял каждую позицию ровно один раз.
Используй existing supervision/locks/sanitized environment из v1 через imports.
Не копируй новый независимый process killer без необходимости.

Состояния: prepared → started до spawn → terminal. Atomic JSON и terminal seal.
Resume запускает только ни разу не начатые arm; без автоматического retrain.
При timeout/failure с неизвестным cleanup останови серию и сохрани censored.
Процессы идентифицируются PID + create_time; чужие процессы не завершать.
Нет process inventory/RSS/cleanup — это unknown, а не ноль и не успех.
Configured cap и фактически наблюдённые CPU/RAM указываются отдельно.
Недостаточная доступная память либо ненаблюдаемый namespace блокируют реальные fit;
не продолжай в известном 8 GiB контейнере с притворным лимитом 24 GiB.

## Evidence, re-audit и отчёт

Для каждого arm сохраняй точный current-run pointer, без поиска «последнего» glob.
Проверь hash отчёта, inputs, config, source, CSV, output IDs/order/schema,
literal classes/probability semantics и независимо пересчитай внешнюю метрику.
Сохрани provenance обучения и доказательство полного TRAIN refit.
Для v3 переиспользуй строгую receipt-chain проверку существующей лаборатории.
Для direct AutoML нужен отдельный typed audit его native checkpoint/phase artifacts.
`qualify_unseen_tasks.inspect_model_evidence` полезен как образец, но его glob-поиск
нельзя выдавать за точную привязку к единственному current run.
Для fixed-arm нужны frozen recipe/source, exact phase inputs, output hashes,
early-stopping evidence и подтверждённый свежий full refit.
Не имитируй v3 `run.json` для методов, которые по этому протоколу не исполнялись.

Re-audit выполняет только чтение и пересчёт: без fit/download/исправления evidence.
Изменение source/input/labels/CSV/config/receipt после запуска должно обнаруживаться.
Prior fallback отражается отдельно и не получает model-quality admission.
Успешный CSV с неполными resource observations остаётся provisional.

В JSON/Markdown покажи все 4 cases и 16 arm, включая missing/failed/censored/skipped.
Основные сравнения: каждый из трёх методов против `full_agent_local` на том же case.
У каждого сравнения denominator — 4 запланированных cases; пропуски не удалять.
Покажи raw metric/direction и signed difference; разные метрики не усреднять.
Покажи wall time, cleanup, peak RSS/CPU observation, CSV/full-refit/API status.
Ноль API подтверждается фактическими наблюдениями соответствующего метода;
отсутствие ключей в env само по себе доказательством не является.
Полезные диагностические observations: время startup, startup AutoML, local search,
outer/refit и размеры fit-данных там, где native path позволяет это наблюдать.
Observer не меняет параметры/очереди/ветвление, его время входит в бюджет.
Победа на 360 s не доказывает победу на 10 500 s или рост contest score.

## Этапы и приёмка

**A — обязательно, без настоящих ML fit:** протокольные fixture/unit tests,
CLI estimate/preflight, точные pointers, tamper detection, бюджет/resume/cleanup,
failures и report arithmetic. Fixture — механизм, не качество модели.
Проверь Windows/Linux paths, spaces/Unicode, spawn и сохранение оригинальной ошибки.

**B — реальные четыре cases, только при подходящем наблюдаемом CPU/RAM:**
один заранее sealed план на все 16 arm, без выборочных повторов после неудачи.
Если среда непригодна, остановись с честным blocked/censored/no-fit результатом
и готовыми командами для Windows ПК. Не арендуй мощности и не скрывай пропуски.

PR должен содержать код, воспроизводимые Windows PowerShell/Linux команды,
точный freeze/config, тестовые результаты, компактный execution ledger,
ошибки/ограничения и hashes результатов. Raw data/labels/models/logs — вне Git.
Проверка production archive должна подтвердить отсутствие ablation/data/models;
проверка сборки не означает новую отправку решения.
Не заявляй реальные ML-прогоны, Windows series, наблюдение ресурсов или выигрыш,
если есть только mocks/tests/preflight. Это допустимый честный промежуточный итог.
