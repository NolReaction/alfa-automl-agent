# Автоматическое предложение одной схемы истории

Экспериментальная опция `--history-auto` соединяет ограниченный schema-proposer
с принятым history runtime. Без неё маршруты и defaults сохраняются. OOF-калибровка
и research policy этим изменением не подключаются.

```powershell
.\.venv\Scripts\python.exe solution.py task.md --mode offline --time-budget 90 --history-auto
```

`offline` не создаёт API-клиент и не читает `.env`. С `--mode llm` та же
дополнительная host-попытка предшествует обычному исследовательскому диалогу.
Опция несовместима с `--history-config`, `--history-weight`, `--validation` и
`--time-column`: применяется уже выбранная v3 схема проверки, вес всегда 0.5.
Имена колонок и частей ключа вручную не задаются.

## Допуск и данные

Discovery разрешён только для регрессии с открытым INNER-исследованием,
измеренным frozen baseline, непустым INNER и временными outer/INNER с одной
и той же разрешённой колонкой периода. Сохраняются task-semantics guard и
измеренный резерв на outer/refit. IID и классификация discovery пропускают.

Parent сериализует **только spec.features строк gate.fit_rows в research_train**.
CSV discovery не содержит технический ID, target, INNER-held, outer или test.
Хеш literal fit IDs остаётся у host отдельно. Это обычная изоляция subprocess,
а не операционная песочница. Proposer и screening не получают ответы y.

Подготовка CSV, импорты и discovery worker используют одну дополнительную
allowance до 8 секунд из исходного дедлайна. Поздний результат отбрасывается;
штатная остановка дерева процессов может потребовать дополнительного cleanup.
Настроенный cap worker — 1 GiB. Факт измерения RSS/исполнения memory cap зависит
от доступности process inventory и показывается отдельно в execution.resources.

## Детерминированное сужение признаков

Сырые строки не проходят модельный preprocessing. Числовые части ключа берутся
только из declared numeric features и сравниваются точным Decimal; текстовые —
после strip, с сохранением буквальных `NA`, `0007` и `7`. Пустая строка/None
означает отсутствие; числовой ноль не объявляется пропуском по догадке.

Период исключается из частей ключа. Для каждой другой колонки считаются значения
и периоды их появления. Исключаются константы, полностью уникальные поля и поля
без повторения хотя бы одного значения между периодами. Колонка не исключается
только из-за неоднозначности: две грубые части могут образовать полезную пару.

Фиксированный ранг: clean_recurrent_rows по убыванию, затем все
cross_period_recurrent_rows по убыванию, затем имя колонки. Clean — сумма
`distinct_periods - 1` для значений, которые встречаются ровно один раз в каждом
своём периоде. Все recurrent rows включают и самый ранний период значения,
появившегося хотя бы в двух периодах. Это структурный screening, не ML-метрика.

Из ранга берутся максимум 12 частей. Единственный вызов public proposer
использует `max_proposals=1`, не более 48 полных проверок, auto-ключ шириной до
трёх частей и неизменную политику `past_unique_only`. При `limited`, включая
неполный поиск `check_limit`, v1 не активирует даже имеющееся предложение.

Лимиты helper: 250 000 строк, 128 признаков, 8 млн запланированных посещений
ячеек (валидация + screening + консервативная оценка proposer), 64 MiB сырых
UTF-8 значений, 512 символов на значение. Parent отдельно ограничивает
сериализованный CSV 64 MiB. При превышении — явный limited, без первых строк,
случайной подвыборки или частичного запуска кандидата.

## Один кандидат и обычное принятие

Полное top-one предложение проверяется как explicit config. `identity_verified`
остаётся false: повторение значений не доказывает, что это один объект.
Структурное покрытие не означает accepted или выигрыш метрики.

Измеряется один history-кандидат с весом 0.5 через существующий
`research_configured_history`/`_assess_candidate`. Рецепт, включая схему и вес,
фиксируется до outer. Outer и полный refit используют тот же source/recipe;
история строится только из fit соответствующей фазы. Отказ, ошибка или нехватка
времени не заменяют baseline. Нарушения целостности исходников/данных/baseline
не маскируются как безобидный отказ discovery.

Автоматический поиск шириной три не обязан найти известный ручной ключ из шести
частей. Имена публичных колонок и специальный fallback под датасет в runtime
не добавлены. После discovery продолжается прежний LLM или локальный поиск;
повторного временного бюджета он не получает.

## Отчёт и воспроизведение

`run.json.history_discovery` фиксирует policy `auto_schema_v1`, scope,
selected/skipped/abstained/limited/failed и причину, fit/input/source hashes,
выбранную схему, работу screening/proposer, elapsed и execution.resources.
LF source text hash и физический hash worker script записаны отдельно для
совместимости Windows/Linux. `history_research` отдельно сообщает реальное
INNER-измерение кандидата; publication отдельно доказывает конечный CSV.
Selected schema сама по себе не является успешным обучением или публикацией.

```bash
python experiment_history_runtime.py preflight --solver-root /path/to/frozen --history-mode auto
python experiment_history_runtime.py prepare --solver-root /path/to/frozen --out-dir data/experiments/history_auto/v1 --history-mode auto --time-budget 90 --memory-mib 6144 --total-seconds 1800
python experiment_history_runtime.py run --out-dir data/experiments/history_auto/v1 --stage development
python experiment_history_runtime.py audit --out-dir data/experiments/history_auto/v1 --stage development
python experiment_history_runtime.py freeze-confirmation --out-dir data/experiments/history_auto/v1
python experiment_history_runtime.py run --out-dir data/experiments/history_auto/v1 --stage confirmation
python experiment_history_runtime.py audit --out-dir data/experiments/history_auto/v1 --stage confirmation
```

Сравнение: native v3 local_only, control против auto, одинаковый wall-time,
legacy search, без API. Development — четыре синтетических случая seed 313;
confirmation — signal/collision seeds 8117 и 16417, неизменный метод до запуска.
Ответы audit находятся вне task directories. По каждому плечу проверяются
настоящие discovery inputs, INNER feedback, outer/refit receipts и итоговый CSV.
Known public TRAIN sanity идёт отдельно. Такие ограниченные опыты не являются
private score, проверкой Qwen или квалификацией трёх часов на платформе.
