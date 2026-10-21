# ТЗ второму ИИ: независимая лаборатория проверки AutoML

Репозиторий: https://github.com/NolReaction/alfa-automl-agent

## Проверенные стартовые источники данных

Это источники для разработки лаборатории, а не готовая независимая оценка нашей
политики. Выбери точные файлы/версии, проверь лицензию и закрепи хеши до обучения.
Расширь набор до 8–12 разнообразных задач; подготовка лаборатории не должна
зависеть от регистрации на Kaggle или передачи пользователем ключей.

| Источник | Направление | Официальная карточка |
| --- | --- | --- |
| Bank Marketing | бинарная классификация, смешанные признаки | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Dry Bean | многоклассовая классификация | https://archive.ics.uci.edu/dataset/602/dry+bean |
| Wine Quality | регрессия, заранее зафиксировать red/white и metric | https://archive.ics.uci.edu/dataset/186/wine+quality |
| Energy Efficiency | регрессия, один target; второй target исключить из features | https://archive.ics.uci.edu/dataset/242/energy+efficiency |
| Bike Sharing | временная регрессия, явно проверить компоненты счётчика и утечки | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| OpenML benchmark suites | дополнительные готовые задачи и описанные splits | https://openml.github.io/openml-python/latest/examples/Basics/simple_suites_tutorial/ |

Bank Marketing содержит несколько вариантов таблицы и признаки, доступные после
звонка: сначала определи момент прогноза и допустимые features. Bike Sharing
нужно проверять на target-derived columns; порядок времени нельзя заменять
случайным split ради более красивой метрики. Разные targets/варианты одного
источника нельзя считать независимыми датасетами confirmation.

## Задача и результат

Работаем над `NolReaction/alfa-automl-agent` для хакатона. Нужен воспроизводимый
стенд на настоящих публичных табличных данных, который сравнит текущий агент
с последующими улучшениями. Основной разработчик параллельно меняет политику
выбора финалистов. Твоя зона — данные, запуск, независимый аудит и доказательства,
а не ещё одна реализация AutoML или изменение конкурсного runtime.

Рабочий компьютер: Windows 10, Intel i9-10900KF, 32 GB RAM, RTX 3080. Основной
контур проверки — CPU с максимумом 8 вычислительных потоков. GPU можно отдельно
использовать для исследовательских гипотез, но конкурсную пригодность подтверждает
только CPU-прогон в официально совместимой среде. Сначала используем имеющийся
компьютер. Аренда сервера, платные API и другие расходы запрещены без отдельного
прямого указания пользователя.

Нужны два явно различимых результата:

1. Рабочий проверенный **инструмент**: загрузка и подготовка данных, фиксация
   плана, попарные запуски, безопасное продолжение серии, независимый пересчёт,
   краткий отчёт и инструкции под Windows/Linux.
2. Реально собранные **измерения**: сначала небольшой pilot; затем только те
   серии, которые действительно были выполнены. Написанный runner, fake-worker
   tests или подготовленный manifest не считаются завершённым ML-экспериментом.

Не обещай рост конкурсного балла по локальным метрикам. Отрицательные результаты,
timeout, ошибки, пропуски и недоказанная очистка процессов — полезные результаты,
они должны оставаться в отчёте.

## Ветка, база и границы владения

Создай `work/external-benchmark-lab-v1` от точной базы
`a45861d7e5588288db66f9ae088a9726f9597370` (`a45861d`). Это incumbent для первого
сравнения. Не подменяй его новым `main` после начала опыта.

Ты можешь создавать и менять только:

- `research_lab/**`: runner, dataset adapters, manifest, локальные конфигурации,
  схемы evidence, небольшие вспомогательные скрипты и собственный `.gitignore`;
- `tests/test_research_lab_*.py`: meaningful tests лаборатории;
- `docs/research_lab.md`: установка, команды, формат результатов и ограничения.

Не меняй `agent/**`, `solution.py`, `offline_baseline.py`, `validation.py`,
`app/**`, существующие benchmark/qualification modules, корневые зависимости,
`requirements.platform.txt`, конкурсный сборщик и настройки CI. Не меняй `main`,
не сливай PR автоматически, не делай force-push. Публикуй только свою ветку и PR,
если доступ к репозиторию уже дан. При необходимости runtime-исправления пришли
основному разработчику точное воспроизведение; не исправляй его в этой ветке.

Разделяй код и локальные артефакты. Большие CSV, ответы audit, модели, predictions,
логи и ZIP с данными не коммить. Используй уже игнорируемый `data/` либо внешний
каталог. В `research_lab/` допустимы маленькие статические manifests, fixtures и
сводки без исходных данных/секретов. Если нужен новый ignore, размещай его внутри
своей директории. Конкурсный ZIP не должен получить данные, модели или laboratory
модули; проверь это read-only через штатный сборщик.

## Сначала изучи существующие механизмы

Прочитай, затем составь короткую карту повторного использования:

- `benchmark_maximize.py` и `docs/maximize_benchmark_protocol.md`: sealed paired
  plan, источник конкретного `run.json`, receipts, одинаковые бюджеты,
  `provisional_resource_observation`, остановка при censored cleanup;
- `qualify_unseen_tasks.py`, `unseen_cases.py`,
  `tests/test_unseen_qualification.py`: идентичность solver checkout, изолированный
  CLI, санитарное окружение, независимый CSV-аудит, process cleanup и re-audit;
- `benchmark_agent.py`, `benchmark_cases.py`: provenance и проверка публикации;
- `agent/task_spec.py`, `agent/metrics.py`, `agent/resources.py`,
  `agent/execution.py`: контракт задачи, точные метки, метрики и ресурсы;
- `docs/frequency_candidate_protocol.md`: граница development/confirmation и
  сохранение проигрышей вместо выборочного отчёта.

Используй готовые функции там, где их контракт подходит, либо тонкие adapters
в своей директории. Не копируй весь старый runner и не строй альтернативный
solver. Аудитор не должен импортировать Python-код проверяемого candidate root
в свой процесс: кандидат исполняется только дочерним процессом. Если host reuse
импортирует общие модули, закрепи отдельный доверенный auditor root и его хеши.

Учти реальные ограничения старых CLI. Например, `qualify_unseen_tasks.py` имеет
собственный максимум `total_seconds=7200`, а `benchmark_maximize.py` предназначен
для известных синтетических cases и бюджетов больше 300 секунд. Нельзя считать,
что передача другого имени датасета автоматически добавляет real-data suite или
разрешает 10 500 секунд. Недостающее реализуй своим bounded wrapper без изменения
старых guards и без заявления, что старый инструмент уже поддерживает новый режим.

## Публичные данные и manifest

Собери каталог 8–12 настоящих datasets из официальных источников: страницы
владельцев/авторов, официальные UCI/OpenML records либо официальные архивы
исследовательских наборов. Исходный список источников может прислать основной
разработчик; каждую запись всё равно проверь перед скачиванием. Не придумывай
URL, license, checksum или размер. Если доступна только непроверенная ссылка,
запиши `unverified` и не включай её в исполняемый план.

Сначала выбери 2 регрессии и 2 классификации для development pilot. В дальнейшем
нужны разные свойства: числовые и смешанные признаки, категориальные признаки,
дисбаланс, multiclass, missing values; при наличии достоверных group/time полей
добавь задачи с групповой и временной проверкой. Разные перестановки одной таблицы
и несколько target одного источника не выдавай за независимые datasets. Известные
в этом проекте public TRAIN и синтетические генераторы учитывай отдельными
разделами; они не становятся новой независимой confirmation из-за нового seed.

Manifest должен содержать минимум:

| Поле | Смысл |
| --- | --- |
| `dataset_id`, `version`, `source_record` | Стабильное имя, версия и официальный record |
| `source_url`, `download_url`, `retrieved_at_utc` | Проверяемое происхождение и время получения |
| `license`, `license_url`, `attribution` | Реальная лицензия и требуемая атрибуция; неизвестное явно отмечено |
| `source_sha256`, `source_bytes`, `member_sha256` | Хеш исходного файла и используемых файлов архива |
| `adapter_version`, `adapter_sha256` | Закреплённая логика преобразования |
| `target`, `task_type`, `metric`, `direction` | Что предсказываем и как оцениваем |
| `id_rule`, `label_rule`, `positive_class` | Точные ID/метки и семантика probability output |
| `feature_exclusions`, `group_rule`, `time_rule` | Исключённые утечки и основания для split |
| `split_rule`, `seeds`, `row_counts`, `split_hashes` | Воспроизводимые train/audit индексы до обучения |
| `role`, `prior_exposure` | development/confirmation и что уже видела разработка |
| `status`, `reason` | ready/unverified/unavailable/ineligible, без тихого удаления записи |

При первом скачивании считай SHA-256 фактических bytes; если publisher публикует
checksum, дополнительно проверь его. Не называй самостоятельно полученный хеш
официальной подписью источника. Cache адресуй по содержимому, не перезаписывай
готовые версии. Ограничь размер/timeout загрузки; проверяй path traversal, symlink
и распакованный размер архивов. Данные с login wall, непонятной лицензией или
неподтверждённым происхождением пропускай с причиной. Не собирай личные данные
из сторонних сервисов и не используй конкурсные test targets.

Скачивание — только отдельная development-команда. Во время запуска solver и
re-audit сетевых запросов, `pip install`, запросов LLM и внешних данных быть не
должно. Наличие cache проверяется до начала серии. Повреждённый cache блокирует
опыт, а не автоматически скачивает новую версию в середине измерения.

## Контракт подготовки и защита от leakage

Для каждого случая создай обычную задачу агента: `task.md`, `train.csv`,
`test.csv`, `data_dictionary.csv`, `sample_submission.csv` в поддерживаемой
структуре. Используй существующий TaskSpec и проверяй контракт до дорогих fit.
Дай solver только train labels, audit/test features, описание и template.

Внешние audit labels хранятся отдельно от каталога задачи, не передаются в argv,
env, task text, sample submission или driver. Target и алгебраически производные
от него поля исключаются из features по заранее описанному adapter rule.
Недопустимо кодировать audit truth в ID или переиспользовать original row index
как случайный числовой признак. ID остаётся точным идентификатором.

Обучаемые преобразования — imputation, category dictionaries, frequency,
scaling, feature selection и любые target-derived statistics — обучаются только
на текущем fit. Их нельзя предварительно обучить на всей таблице в adapter.
Статические schema rules и механическое переименование столбцов допустимы,
если они не используют audit targets. Audit/test targets удаляются до создания
каталога задачи и не становятся доступными в scratch-копии исходного CSV solver.

Разбиение закрепляется до измерений:

- обычные iid задачи: заранее объявленный random/stratified split;
- группы, сущности и exact/near duplicate rows: не допускай пересечение одной
  единицы между train и audit, документируй определение дублей;
- временные задачи: train только из прошлого, audit позже, без случайной
  перестановки периода; при необходимости заранее заданный gap;
- несколько seeds меняют допустимый split или модельную случайность по
  manifest, не используются для поиска красивого audit результата.

Для классификации сохрани literal strings (`01`, `NA`, пустая строка там, где
это реальная метка), class order и смысл positive class. Нельзя делать скрытое
`astype(int)` меток или считать строки `NA` пропусками. Для ROC AUC/LogLoss нужен
корректный probability contract; unsupported class/metric case получает явный
skip, а не незаметно переключается на Accuracy. CSV проверяется по сохранённым
bytes: число строк, ID/order, столбцы, конечность чисел, допустимые классы и
вероятности, отсутствие target leakage.

## Confirmation и обмен с ИИ, который меняет политику

Ты — инженер evaluator/harness. Основной ИИ — разработчик policy. Не используй
внешние confirmation labels/метрики для изменения policy и не отправляй их
разработчику policy до заморозки проверяемого кода и конфигурации.

Практический порядок:

1. Pilot development данные и их результаты можно обсуждать. Заранее пометь
   их известными для разработки; повторная проверка на них остаётся development.
2. До confirmation зафиксируй набор/версии данных, adapters, split IDs, seeds,
   порядок плеч, budgets, метрики и критерий сравнения. В auditor-owned каталоге
   оставь labels; разработчику policy передай только manifest без targets,
   fingerprints и объём запланированного опыта.
3. Основной разработчик передаёт точный чистый candidate commit/config и
   подтверждает freeze. Закрепи его вместе с incumbent и hash auditor method.
   До этого этапа не раскрывай per-case confirmation scores, per-row errors,
   target statistics или hints по слабым случаям.
4. Выполни замороженную серию. После получения confirmation результатов не
   изменяй candidate, набор, seed, adapter, метрику или budget ради улучшения
   отчёта. Новая итерация означает новый versioned план и новый confirmation
   material; ранее открытый материал становится известным для разработки.

Разные каталоги и договорённость ролей — граница data flow, а не OS sandbox.
Публичная доступность исходной таблицы также не доказывает, что модель/ИИ никогда
её не видел. Пиши, какая независимость обеспечена процедурой и какие ограничения
остались. Уже показанные цели или результаты нельзя задним числом объявить
невиденными. Если разделение ролей не удалось, явно снизь статус до development.

## Runner: обязательное поведение

Создай cross-platform CLI `python -m research_lab` с командами как минимум
`catalog`, `fetch`, `prepare`, `run`, `resume`, `re-audit`, `report`, `estimate`.
Ниже описан целевой интерфейс, который ещё нужно реализовать, а не существующие
команды репозитория. Запуск без явного `run`/`resume` не должен тренировать.

`prepare` принимает incumbent root, candidate root, interpreter, manifest,
профиль, budgets и seeds. Он только проверяет условия, материализует tasks,
запечатывает plan и печатает число плеч/максимальное время/объём cache.
Используй новую директорию run, атомарные JSON-записи и seals. До первого fit
закрепи source bytes и Git SHA обоих solver roots, dirty status, Python/deps,
OS/CPU/RAM, CLI/config, dataset/split hashes, argv и auditor method.

Сравнивай полный incumbent `a45861d` с точным более поздним candidate checkout,
полученным от основного разработчика. Не меняй кандидатам search defaults,
число trials, startup или ensemble ради ускорения. Если технический smoke
применяет test-only cap, обозначь его отдельно и не выдавай за production quality.
Двум плечам нужны побайтно одинаковые task inputs и один внешний inclusive
wall-clock budget; фиксируй порядок заранее и чередуй его между парами.

Время worker начинается до импортов и охватывает startup, аварийный prior,
preprocessing, поиск, INNER, outer, full refit, publication и запись evidence.
Остаток одного исходного deadline передаётся solver; ошибка не выдаёт новое
время. Supervisor не добавляет скрытый training allowance. Cleanup после
остановки измеряется отдельно. Выход за deadline получает явный outcome.
Учитывай ограничения совместимого выбранного режима: для проверки adaptive
coordinator реальный остаток после импортов должен быть больше 300 секунд.

По умолчанию допускается одна активная ML-задача и 8 потоков на весь запуск.
Не запускай параллельно incumbent/candidate на одном PC. Удали `OPENAI_*`,
proxies и developer import paths из worker env, выстави thread caps для native
библиотек; CPU lane не получает GPU. Проверяй ноль API attempts по actual reports,
а не только отсутствие заданного ключа. На 32 GB PC оставь память ОС: например,
стартовый child RSS budget 24 GiB, с явной записью configured/observed limits.
Не объявляй polling RSS жёсткой OS memory sandbox.

Windows runner обязан учитывать пути с пробелами/кириллицей, UTF-8, `spawn`,
группы процессов, корректное завершение descendants и невозможность удалить
открытые файлы. Не переноси POSIX `killpg` без Windows-ветки. Фиксируй process
identity с create time, не убивай посторонние PID. Сохраняй availability и
ошибки мониторинга; неизвестный RSS/CPU не заменяется нулём.

После failed/timeout/memory-killed arm без подтверждённого cleanup пометь его
`censored` и останови серию до ручной проверки. Успешный CSV с неполным resource
наблюдением допускает только `provisional_resource_observation`. Не пиши
`cleanup_complete=true` просто из-за завершения parent process.

## Evidence и продолжение после обрыва

Каждое плечо получает уникальный run ID и состояния `prepared → started →
finished/failed/censored`. Перед spawn атомарно запиши `started`, argv, UTC и
monotonic origin. После обрыва `resume` может запустить только **никогда не
начатые** плечи, при неизменных seals и доказанной очистке прежних процессов.
Начатое плечо автоматически не переобучается. Сохрани его `interrupted`/`censored`
evidence; для повторения нужен новый явно названный run/plan, который не стирает
первый исход. Lock должен исключать два одновременных coordinator одного run.

Пиши metadata, stdout/stderr bounded tails, exit code, deadline/timeout/memory,
API attempts, observed resources, cleanup, input/source hashes до и после,
точный путь текущего `run.json` и его SHA, selected recipe/source hash,
outer/full-refit receipts, финальный CSV hash и его independent audit score.
Не выбирай отчёт через «последний glob» и не принимай старый `submission.csv`
за новый успех. Успешная публикация должна соответствовать текущему refit и
его frozen recipe; fallback/prior сохраняй с отдельным quality-unverified
статусом, а не как победу качественной модели.

`re-audit` читает только сохранённые материалы, ничего не обучает, не скачивает
и не чинит. Он обнаруживает изменённые labels, inputs, code/config, CSV и
receipts. Если в план вошли короткие и длинные budget tiers, сохраняй их
раздельными: нельзя объединить их в одну честную попарную оценку.

## Бюджеты и последовательность работ

| Этап | Объём | Назначение |
| --- | --- | --- |
| A: инструмент | Tests, dry-run, tiny fixture smoke | Проверить механизм, без заявления о качестве |
| B: development pilot | 2 регрессии + 2 классификации, 1 seed, 2 плеча, обычно 360 s/плечо | Проверить реальные данные, стоимость и adaptive route |
| C: bounded suite | 8–12 datasets, 2–3 seeds, одинаковый заранее выбранный tier | Сравнить замороженные версии, сохранить все исходы |
| D: финальная проверка | Заранее выбранные представительные случаи, 10 500 s/плечо | Проверить длинный путь в совместимой CPU-среде |

Не запускай этап C/D автоматически вслед за установкой. Сначала покажи оценку
числа запусков и верхней границы времени, затем используй явно указанный профиль
в рамках рабочего задания. Четыре pilot cases × один seed × два плеча × 360 s —
48 минут только верхнего training budget, плюс явно учтённые подготовка/cleanup.
Десять datasets × три seeds × два плеча × 900 s — 15 часов; не называй такую
серию «быстрой». Два финальных случая × два плеча × 10 500 s — около 11 ч 40 мин
только budget. Заранее сообщай это пользователю в summary.

Начни с development pilot, если sources/cache и локальная среда доступны.
Measured pilot timings могут обосновать новый заранее объявленный tier,
например 360/900/1800 секунд. Изменение tier требует нового плана до просмотра
confirmation scores; оба плеча одной пары получают одинаковые лимиты. Если
не хватает ресурса/времени, честно закончи pilot и подготовь следующие команды,
не выдавая отложенную серию за выполненную.

`estimate --dry-run` должен показывать wall-clock upper bound, ожидаемый
cache/disk объём, RAM/CPU требования и число запусков. Локальный direct rental
cost — 0; стоимость электричества не придумывай. Для аренды можно показать
формулу `hours × user_supplied_hourly_rate` и требуемую конфигурацию. Не выбирай
платный тариф, не вводи карту, не арендуй и не расходуй API-бюджет самостоятельно.

## Отдельный GPU exploratory lane

Добавь opt-in `research_lab` режим для небольших CatBoost/XGBoost GPU опытов на
RTX 3080 только после CPU pilot. Он не меняет конкурсный solver и не входит в
CPU qualification. Если библиотека/driver/CUDA отсутствуют, сохрани явный skip;
не устанавливай другие версии core dependencies и не скачивай pretrained models.
Дополнительные development dependencies держи отдельно, с точными версиями и
инструкцией установки по явному выбору пользователя.

GPU lane имеет собственные budget, seed, task inputs, candidate config,
device/driver/library versions и evidence. Метрики помечаются `exploratory_gpu`.
Выигрыш GPU не означает, что это можно запустить в конкурсе с 8 CPU. Перенос
гипотезы в runtime делает основной разработчик, после чего нужна новая парная
CPU-проверка: точный допустимый code/config, те же метрики/данные и одинаковый
внешний budget. Без неё GPU-вывод остаётся provisional.

## Meaningful tests и критерии приёмки

Нужны быстрые tests без реальных LLM/скачиваний/долгих fit. Проверь как минимум:

- изменённые source/input/manifest/split hashes блокируют исполнение/re-audit;
- labels и audit paths отсутствуют в task, argv/env и generated driver;
- IDs, literal labels и probability class order сохраняются точно;
- group/time split и target-derived exclusions работают на намеренно
  конфликтных fixtures, а не только на удобном happy path;
- timeout включает imports/finalization; второй budget после ошибки не выдаётся;
- после started/run interruption нет автоматического повторного fit, а resume
  запускает только untouched arms; два coordinator блокируются;
- stale CSV, чужой run.json, подменённый receipt и неполный refit не проходят;
- timeout/ошибка + unknown cleanup дают censored/stop, недоступный мониторинг
  не превращается в нулевое потребление;
- все проигрыши/ошибки/skip остаются в JSON/Markdown и denominator summary;
- путь с пробелами/Unicode, sanitized env и Windows cleanup проверяются
  подходящими unit tests; факт реального Windows исполнения указывается отдельно.

Сначала запусти собственные tests, затем маленький native smoke, затем pilot.
Не повторяй весь core suite после каждой правки prose; перед PR выполни
уместные integration checks и убедись, что изменения ограничены своей зоной.
Проверка production archive должна показать отсутствие `research_lab`, данных
и моделей; генерация такого архива не означает отправку на платформу.

## Целевые команды для README лаборатории

Реализуй и проверь этот интерфейс либо документируй эквивалентные названия.
`candidate` здесь — отдельный checkout точного freeze от основного разработчика;
не назначай candidate случайной текущей веткой. До получения freeze можно
использовать два checkout incumbent для проверки равенства механизма, но
это smoke, а не evidence улучшения.

PowerShell, из корня клона репозитория:

```powershell
git switch -c work/external-benchmark-lab-v1 a45861d7e5588288db66f9ae088a9726f9597370
git worktree add --detach ..\alfa-incumbent a45861d7e5588288db66f9ae088a9726f9597370
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_research_lab_*.py" -v
.\.venv\Scripts\python.exe -m research_lab catalog --manifest research_lab/datasets.json
.\.venv\Scripts\python.exe -m research_lab fetch --manifest research_lab/datasets.json --profile pilot --cache data/lab-cache
.\.venv\Scripts\python.exe -m research_lab estimate --profile pilot --dry-run
.\.venv\Scripts\python.exe -m research_lab prepare --manifest research_lab/datasets.json --profile pilot --incumbent-root ..\alfa-incumbent --candidate-root ..\alfa-candidate --python .venv\Scripts\python.exe --cache data/lab-cache --output data/lab-runs/pilot-001 --time-budget 360 --threads 8
.\.venv\Scripts\python.exe -m research_lab run --output data/lab-runs/pilot-001
.\.venv\Scripts\python.exe -m research_lab resume --output data/lab-runs/pilot-001
.\.venv\Scripts\python.exe -m research_lab re-audit --output data/lab-runs/pilot-001
.\.venv\Scripts\python.exe -m research_lab report --output data/lab-runs/pilot-001
```

Зависимости лаборатории устанавливаются один раз в development-среду. Не делай
`pip install` из runner или solver. Если нужны packages из полного официального
списка, перечисли конкретную причину и поддерживаемый вариант установки;
не переписывай официальный список. Выбор AutoGluon/его отсутствие должен быть
одинаковым в двух CPU плечах и отражаться в versions/skip evidence.

Bash, после создания аналогичных worktrees и venv:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_research_lab_*.py' -v
.venv/bin/python -m research_lab estimate --profile pilot --dry-run
.venv/bin/python -m research_lab prepare --manifest research_lab/datasets.json --profile pilot --incumbent-root ../alfa-incumbent --candidate-root ../alfa-candidate --python .venv/bin/python --cache data/lab-cache --output data/lab-runs/pilot-001 --time-budget 360 --threads 8
.venv/bin/python -m research_lab run --output data/lab-runs/pilot-001
.venv/bin/python -m research_lab re-audit --output data/lab-runs/pilot-001
```

Не запускай `resume` механически, пока не прочитан статус: unknown cleanup
требует ручной проверки, а уже начатые arms не перезапускаются. Повторный
`re-audit` не имеет права менять исходные receipts или обучать новую модель.

## Что вернуть основному разработчику и Диме

В PR опиши конкретное поведение инструмента, выполненные проверки и ограничения.
Отдельно перечисли фактически завершённые этапы A/B/C/D. Итог должен содержать:

1. Ссылку на свою ветку/PR, точный commit лаборатории, список изменённых файлов
   и команды установки/запуска PowerShell. Не делай merge самостоятельно.
2. Маленький manifest с проверяемыми provenance/license URLs, source/adapter/
   split SHA-256, версиями данных, флагом prior exposure и development/confirmation.
3. Environment summary: Python, все задействованные packages, OS, CPU/RAM,
   наблюдаемый thread cap, configured/observed memory, GPU отдельно.
4. Точные incumbent/candidate commits и source hashes, plan/method/config hashes,
   время начала/окончания и фактически потраченное время; API requests = 0,
   если это подтверждено actual reports.
5. Компактную таблицу: dataset/seed/tier, metric/direction, incumbent/candidate,
   signed difference, seconds, finalization/CSV/cleanup status, outcome/reason.
   Разные исходные метрики не усредняй в бессмысленный общий score. Если даёшь
   агрегат ranks/win-rate, заранее объяви правило и покажи denominator, ties,
   failures/censored/missing отдельно; не исключай проигрыши молча.
6. Сводку wins/ties/losses/failures/censored, существенные регрессии, что
   подтверждено CPU, что только GPU exploration, что пока не выполнено.
7. Hash компактного отчёта и manifest; путь к локальному полному evidence bundle.
   Дима может передать summary + manifest + versions + hashes без огромных CSV.
   Большой evidence bundle передаётся отдельным файлом при необходимости,
   в Git остаются только код, небольшой отчёт и проверяемые fingerprints.

До freeze передавай только development результаты и confirmation plan/hashes.
После freeze можно отдавать весь компактный итог. Не прячь неудачу за статусом
«инструмент готов»: отдельно напиши, сколько настоящих пар реально завершилось.

Начинай с чтения существующих механизмов, реализации минимального полезного
интерфейса и четырёх development cases. Если среда не позволяет запустить fit,
всё равно заверши проверенный инструмент, tests, manifests и воспроизводимые
команды; честно укажи, что сбор измерений ещё предстоит. Не останавливайся на
плане, когда доступный локальный pilot можно действительно выполнить.
