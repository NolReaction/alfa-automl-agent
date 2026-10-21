# Мураду: отдельный CPU XGBoost-кандидат

Дата: 21 октября 2026, МСК. Координация: Дима.

## Результат

Добавить готовый к подключению XGBoost-кандидат для регрессии, бинарной и
многоклассовой классификации: код, проверки контракта, настоящий subprocess
smoke и PR. Это новый самостоятельный движок. История, генератор схем и их
ускорение уже сделаны; повторять эти задания не требуется.

`requirements.platform.txt` уже разрешает `xgboost==2.1.4`. Сейчас отдельного
XGBoost-кандидата нет; наш AutoGluon использует GBM/CAT/RF/XT/LR. Добавление
движка не означает доказанный прирост качества или включение по умолчанию.

## База и владение

Репозиторий: https://github.com/NolReaction/alfa-automl-agent.
Точная база: `184408390c83ef951e5a570cf9d2d6c3e9e72f98`.
Ветка: `work/murad-xgboost-v1`. Не продолжай старую history-ветку.

```powershell
git status --short
git fetch origin work/finalist-selection-v1
git worktree add ..\alfa-xgboost -b work/murad-xgboost-v1 184408390c83ef951e5a570cf9d2d6c3e9e72f98
```

Проверяй exit code. Существующие ветки, worktree и чужие изменения сохраняй.

Разрешены только новые файлы:

- `agent/xgboost_candidate.py`
- `xgboost_candidate_cases.py`
- `experiment_xgboost_candidate.py`
- `tests/test_xgboost_candidate.py`
- `tests/test_xgboost_candidate_experiment.py`
- `docs/xgboost_candidate_report.md`

Общий runner, search, model_config/models, существующие history/boosting-модули,
CLI, app, builder, requirements, CI и `research_lab/**` не меняй. Подключение
к общему поиску и приёмка остаются у Димы и основного ИИ. Второй внешний ИИ
делает лабораторию реальных датасетов; её код и данные не дублируй.

Прочитай `agent/boosting_candidate.py`, `agent/research_candidate.py`,
`agent/features.py`, `agent/models.py`, `agent/metrics.py` и
`tests/test_boosting_candidate.py`. Используй действующий v3-контракт.

## Интерфейс

```python
def make_xgboost_recipe(task_type, metric_name, *, validation,
                        parameters=None, feature_config=None,
                        max_iterations=2000, stopping_rounds=50):
    ...

def execute_xgboost_candidate(X_fit, y_fit, X_predict, context, *, return_evidence=False):
    ...

def fit_predict(X_fit, y_fit, X_predict, context):
    ...

def candidate_source():
    ...
```

Recipe — нормализованный конечный JSON, version=1, family=`xgboost_native_v1`.
Он фиксирует задачу, метрику, параметры, preprocessing, внутреннюю политику split
и предел итераций. Входы не мутируются. Неверные/неизвестные поля и bool вместо
числового ограничения отвергаются до обучения. Не полагайся на глобальное
состояние или настройки старого запуска. Проверяй фактическую версию/API 2.1.4
по установленному пакету и официальной документации перед реализацией.

`candidate_source()` возвращает обычный код `fit_predict` для freeze/execute
хоста. Обёртка пишет `xgboost_evidence.json` с режимом exclusive-create по
образцу boosting-кандидата. Evidence не является доверенной оценкой качества:
host сам проверяет predictions и считает метрику по своим ответам.

## Обучение и ресурсы

1. CPU, `tree_method="hist"`, `device="cpu"`, максимум 8 потоков, фиксированный
   seed. Строгий whitelist параметров: исключи GPU, пользовательские callbacks,
   произвольные objectives, пути и ресурсы сверх лимитов. Достаточно двух заранее
   фиксированных конфигураций; HPO и большой перебор не нужны.
2. Внутренний early-stopping split строится только из переданных X_fit/y_fit по
   явной host policy. Для времени — строго прошлое, для групп — без пересечений.
   При невозможной политике или недостатке классов — честный skip с причиной.
3. Preprocessing обучается на внутреннем fitting-подмножестве. Категории,
   заполнение пропусков и статистики held/predict не входят в fit. Для v1
   предпочтителен существующий fit-only ordinal preprocessing; оцени пригодность
   `models._encoded_preparation(..., linear=False)`. Сначала обеспечь корректность,
   без усложнения native categorical и sparse-zero semantics.
4. Число деревьев выбирается на внутреннем held. Затем создаются новый
   preprocessor и модель, обучаемые на ВСЕХ переданных X_fit/y_fit с выбранным
   числом деревьев. Predict-ответы не участвуют. Поведение не ветвится по phase:
   INNER, OUTER и full refit исполняют один и тот же замороженный алгоритм.
5. Кодировка классов и порядок probability columns соответствуют host classes,
   включая positive_class, строковые/нечисловые labels и multiclass. Формы,
   конечность и нормировка проверяются. Внутреннюю logloss/RMSE-прокси нельзя
   называть точным Macro F1/AUC/MAE, если оптимизируется другая величина.
6. Используй исходный deadline хоста. Early stopping, fresh refit и prediction
   делят один остаток; ошибка не создаёт новое время. Supervisor хоста остаётся
   окончательным ограничителем, один worker — максимум 1800 s. Оставляй измеренный
   запас на fresh refit и проверяй deadline перед дорогими стадиями.
7. Зафиксируй и проверь caps размерности/preprocessing до больших allocation.
   Выход за caps — явный skip/limited, без скрытого уменьшения данных. Новый
   модуль ничего не скачивает, не обращается к API и не читает `.env`.

Evidence: версия пакета, frozen recipe, split policy и размеры, actual training
iterations, best/selected tree count, критерий остановки и его proxy-статус,
full-fit/predict row counts, фактические времена стадий, deadline outcome,
CPU/thread configuration и skip/failure reason. Не выдавай configured limits
за измеренные пиковые ресурсы или hard sandbox.

## Проверка и передача

Обязательны содержательные проверки:

- regression, binary и multiclass; missing values и новые predict-категории;
- class order/positive_class, отсутствие held-класса, неизвестные параметры;
- внутренний preprocessing не видел held; финальный refit видел все fit-строки;
- time/group boundaries и отсутствие same-period/future leakage;
- изменение predict-batch не меняет fit-статистики/выбор деревьев;
- истёкший бюджет, остановка перед refit, ресурсный skip, неизменные входы;
- exact CSV/probability shapes, сохранённые receipts и независимый host rescore.

Сначала unit tests, затем 3–4 небольших настоящих subprocess smoke последовательно
на новых каталогах с сохранением ошибок/таймаутов. Не заменяй XGBoost mock-моделью
в native smoke. Каталог и конфигурации запиши до запуска. Старые известные
synthetic cases допустимы для механики, но не являются независимым quality audit.
Не заявляй превосходство полного агента по сравнению только с ExtraTrees.

В отчёте: база/head/source hashes, pinned environment, точные команды, таблица
actual status/time/metric/refit/receipts, проверки утечек, ограничения и ноль API.
Unit tests и `--help` не считай выполненным ML-экспериментом. Если native запуск
недоступен, прямо укажи это и оставь готовые команды для Димы; не выдумывай результаты.

Commit/push своей ветки, без merge/force push. Пока PR #27 не принят, открой
свой PR с base `work/finalist-selection-v1`: это stacked PR. После приёмки #27
Дима перенаправит его в main. Не сливай свой PR раньше #27; свою зону изменений
проверяй относительно точной базы `1844083`, а не старого main. Исходные данные,
модели и сырые логи — в игнорируемом data, компактные результаты — в отчёте.
Результат приёмки — безопасный изолированный кандидат, готовый к общей проверке;
рост метрики не является обещанием или условием честного отчёта.
