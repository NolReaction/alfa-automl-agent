# AutoGluon Tabular: ограниченный CPU-кандидат

Модуль `agent/autogluon_candidate.py` предоставляет настоящий кандидат для
существующего research v3, а не отдельный механизм публикации CSV. Его допуск,
внешнюю проверку, полный refit и итоговый файл контролирует обычный host.
Рост качества этим модулем пока не доказан.

```python
from agent.autogluon_candidate import (
    availability, candidate_source, make_autogluon_recipe,
)

recipe = make_autogluon_recipe(
    "classification", "Macro F1",
    validation={"strategy": "grouped"},
    time_limit=180, num_cpus=8, memory_limit=24,
)
source = candidate_source()
```

Host передаёт **разрешённую** политику из `gate.choice`, а не произвольный новый
split. `random`, `grouped` и `time` поддерживаются в тех же сочетаниях с типом
задачи, что `agent.splits`. Время требует явного `time_column`. Нет неявного
перехода с невозможной временной/групповой проверки на случайную.

## Контракт и обучение

- Пакет строго закреплён на `autogluon.tabular==1.3.1`, как в предоставленном
  платформенном requirements. В конкурсном коде нет установки пакетов.
- `candidate_source()` объявляет обычный
  `fit_predict(X_fit, y_fit, X_predict, context)` и импортирует модуль через
  flat-compatible `agent` package.
- На каждом вызове создаются новый predictor и временный каталог внутри
  текущего worker. Состояние модели не переносится между фазами.
- Собственный tuning split получается **только из X_fit/y_fit** существующим
  `split_rows`. Дубли исходных признаков разделяются группами, временная
  проверка выделяет целиком последний доступный fit-период. Все классы должны
  быть представлены в обеих внутренних частях.
- AutoGluon обучается на внутренней fitting части с разрешённой tuning частью,
  затем выполняет `refit_full` выбранной модели на совокупности этих частей.
  Прогноз строится именно `_FULL`-моделью. Из `model_info(refitted)` требуется
  настоящий `num_samples`, равный числу всех переданных fit-строк. Пропущенный,
  ложный или неполный refit отклоняется.
- Код не читает `context.phase` и не подстраивает поведение под inner/outer/full
  refit. Во всех фазах одинаковы рецепт и алгоритм обучения.
- Чужие target и ID не поступают в функцию. Признаки X_predict не входят ни в
  fitting, ни в tuning. Нет внешних данных, скачивания весов, сетевых запросов
  или чтения `.env`.
- Буквальные классы `01`, `NA`, `class c` сохраняются. Полные вероятности
  переставляются в объявленный host порядок. Для label-метрик применяется
  argmax этих вероятностей; калибровка порогов выключена. Binary positive class
  передаётся predictor явно, когда определён host.

CPU-семейства ограничены `GBM`, `CAT`, `RF`, `XT`, `LR`. Нейросетей и foundation
моделей нет. Выключены текстовые n-gram/special/raw и vision признаки,
bagging, stacking, dynamic stacking, Ray parallel fit и weighted ensemble.
Неявные случайные bag-folds поэтому не могут нарушить временную/group policy.

## Ограничения ресурсов и готовность

Допустимы 1–8 CPU, 1–24 GB memory_limit, 15–1800 секунд recipe time_limit.
Дополнительно действуют caps: 250 000 fit/predict строк в каждой части,
256 признаков и 16 миллионов суммарных feature cells. Неподходящий кандидат
отклоняется; прежний результат host сохраняет.

Host может передать `deadline_monotonic` и `time_budget_seconds`. Они только
сокращают локальный срок, не продлевают его. Для первичного AutoGluon fit
отводится 45% остающегося срока; остаток предназначен для настоящего
`refit_full`, prediction и проверки. AutoGluon `time_limit` приблизительный,
`memory_limit` мягкий, а `refit_full` не гарантирует остановку к сроку.
**Жёсткую границу обеспечивает внешний worker timeout/deadline host**, а не
заявленное значение в JSON. После истечения срока прогноз не публикуется.

`availability()` проверяет установленную версию, существующий host namespace
guard и способность настоящего `psutil.Process(os.getpid()).memory_info()`
прочитать метаданные процесса. При несовпадении PID namespace численно
совпадающий посторонний host PID даже не исследуется.
Возможные причины отказа: `package_unavailable`, `version_mismatch`,
`process_inventory_unavailable`. Здесь нет искусственных RSS/CPU значений.
Проверка одного процесса не доказывает соблюдения лимита всей process tree;
за это отвечает внешний resource monitor. `memory_limit_enforced=false` в
worker evidence не выдаётся за доказательство жёсткого cap.

Успешный worker дополнительно сохраняет `autogluon_evidence.json`: внутреннее
разбиение, реальные fit-row counts, имена выбранной и refit-модели, host-clamped
time budget и отдельные wall timings. Evidence не заменяет host recomputation
метрики, проверку CSV или execution receipt.

## Проверки и фактический статус

26 pure contract tests прошли в проверенном project runtime Python 3.12.14.
Они используют fake predictor для проверки границ протокола, а не изображают
результат обучения: literal class order, групповое/временное разделение,
полный refit с настоящим backend row-count field, deadline clamp, отклонение
недоступного process inventory, отсутствие fit на prediction rows и запрет
неподдерживаемых моделей/ресурсов.

Для native-проверки создано отдельное окружение
`/workspace/scratch/271f408d330f/alfa-autogluon-runtime`. Исходное
`alfa-runtime` не изменялось; NumPy 1.26.4, pandas 2.2.3, SciPy 1.12.0,
scikit-learn 1.5.2, CatBoost 1.2.8 и LightGBM 4.6.0 сохранены. В отдельное
окружение установлены четыре AutoGluon-пакета 1.3.1 и их недостающие служебные
зависимости. Факт импорта backend подтверждён, но не заменяет исполнения.

Два локальных native-запуска сохранены в
`data/experiments/autogluon_candidate/native_contract_20261021_v{1,2}`:

1. v1 остановлен до обучения: `label_count_threshold` ошибочно передавался в
   `.fit()`. Для версии 1.3.1 исправлено на constructor
   `learner_kwargs={"label_count_threshold": 1}`; добавлена проверка контракта.
2. v2 дошёл до preprocessing и остановлен **до обучения модели**:
   `AutoGluon ResourceManager` не смог прочитать `psutil.Process()` из-за
   несовпадения текущего PID namespace и смонтированного `/proc`.
   Метаданные и traceback сохранены. Никакие данные process inventory не
   подменялись; нового native rerun с другими seeds не было. Готовность теперь
   явно возвращает `process_inventory_unavailable` в этом окружении.

Следовательно, локальный native результат не объявляется успешным. Для
настоящего Linux/Windows CI с корректным process inventory подготовлен smoke:

```bash
python experiment_autogluon_smoke.py --output data/experiments/autogluon_candidate/native_ci
```

Это ровно два последовательных case: grouped classification и time regression,
каждый с 180 fit-строками, RF-only, 2 CPU, recipe 30 s и host cap 45 s.
Общий host deadline 90 s. Проверяются literal IDs, native receipts, реальный
`refit_full` и fit-row counts. Это проверка исполнения и интерфейса, **не**
парный benchmark качества, сравнение с AutoML или платформенный score.

Официальные API версии 1.3.1:
[fit](https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html),
[refit_full](https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.refit_full.html).
