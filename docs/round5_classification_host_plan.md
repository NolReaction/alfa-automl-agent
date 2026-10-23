# Round 5: заранее заданный план native OFF/ON

**Статус: preregistration до native; результаты и допуск ещё pending.** Helper, CLI и harness реализованы, полный frozen recipe проверен без обучения. До первого большого fit необходимо завершить validation gates ниже, закоммитить код и этот план, затем выполнить `prepare` и отдельный contract smoke. Наличие реализации не считается доказательством достигнутого OOF или качества.

База — `66e9481cb61cb83cfecab53be6984d7d2bf829b3`, ветка `work/round5-oof-host-v1`. Цель — проверить явный opt-in frozen OOF в полном агенте; default остаётся **OFF** при любом результате. Allowlist задания соблюдается; regression rescue-v1, модели, features, guards, splits, supervisor, зависимости и CI не меняются. Свежий main сверяется перед подготовкой; расхождение сохраняется, база посреди пары не меняется.

## Гипотеза и сравнение

`comparison_type = model_plus_scheduling_plus_paid_probe`. Проверяется весь bundle: модель, раннее расписание и оплаченный cost-only probe. Изолированные +0.0151549800 Macro F1 Round 4 не являются ожидаемым или добавочным native gain. Старую allocator-пару с одинаковыми outer 0.6469928093378587 не повторяем.

| Arm | Поведение | Полный native budget |
| --- | --- | ---: |
| OFF, первая | Прежние startup/baseline, порядок ordinary recipes, extension hooks, caps/reserve и lifecycle; probe/OOF отсутствуют | 3 600 с |
| ON, вторая | Ранний frozen OOF вместо ещё не выполненного CB400/depth10/lr0.03; при необходимости один оплаченный identity probe | 3 600 с |

Ровно две большие arms, строго serial, один общий machine lock. OFF должен завершиться и пройти аудит до ON. Нет третьего scheduler-control, новых seeds, повторов, дополнительного model search или расширения бюджета. Суммарный solver budget ≤7 200 с; подготовка, watchdog и cleanup учитываются отдельно и не продлевают solver deadline. Раннее нормальное завершение разрешено.

Главная метрика — независимо пересчитанный **published outer Macro F1 полного агента**. Заранее заданный инженерный порог: **ON − OFF ≥0.002 absolute Macro F1**, при корректных lifecycle, guards и неизменной идентичности. Это одна известная development-пара, не статистическая гарантия и не hidden TEST. Показываются также все классы: support, F1 и paired Δ; новые пороги по увиденным результатам не добавляются.

Если опубликован другой ordinary кандидат, прирост относится к bundle; он не доказывает пользу OOF-модели. `promising_opt_in` требует полной валидной пары, прироста ≥.002 и действительно опубликованного OOF. Если OOF достигнут, но условия не выполнены, статус `rejected`; при отсутствии достигнутого OOF — `not_reached`; невалидная/незавершённая пара — `incomplete`. INNER/feasibility/outer отказы сохраняются отдельно. Отсутствие измерения не считается успехом. Default остаётся OFF, merge и выпуск решает капитан.

## Замороженная идентичность

Единственный OOF recipe создаётся существующим `make_recipe(fixed_incumbent_recipe(), classes, variant="oof")` и проверяется существующим `checked_recipe`. Class order берётся из разрешённого TaskSpec. Ожидаемый полный нормализованный recipe hash:

`cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc`

Несовпадение блокирует native. Нельзя переставлять классы или подбирать поля до совпадения по результату. Полный JSON сохранён в `docs/round5_classification_host_recipe.json`; штатные TaskSpec → factory → `checked_recipe` подтвердили ожидаемый canonical hash без fits. Raw JSON file SHA-256 — `c935a8d23213919130538f2715c7e9f93082f3ef802f670e0d8bfee4d884f8b4`; standalone source hash — `1cb8a3f9df7588b5c4773351b9426c29eedd4c55f297f59fafeeaeefaf300862`. Ordered classes: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms. Эти имена описывают evidence, runtime не использует их как dataset/phase switch. Base LGB400/RF120, веса классов, параметры, seeds, features, OOF/guard options и staging остаются прежними. Identity probe имеет отдельный фиксированный `variant="identity"` recipe, без переноса fitted-состояния.

Проверенные исходные hashes неизменяемых модулей на базе:

| Файл | SHA-256 |
| --- | --- |
| `agent/round4_classification_candidate.py` | `493af581ef9d848ff0036e2a748961d6b097900ffecef3ad2a08316c259256f4` |
| `agent/round4_classification_probability.py` | `e6dca9853fee191c32186a10ec088be8d85fa365f899dcc276dfdf48a1613924` |
| `agent/round4_classification_features.py` | `39ecdd485001549a5c22dc9f7308b36020454737e6b428f33dd4440a56d2c577` |
| `agent/oof_classification.py` | `cc16f7ac683e1ee6c1ab0bbbf4bb575a64cd7cf9e7fa7616a8ebdcd511aafffc` |
| `agent/classification_decision.py` | `e81dc5ca13483ea55dc5f5d3798868fe643e4d64d01bbe82400ee54698cded30` |

Run manifest дополнительно фиксирует фактические hashes всех импортируемых runtime dependencies, source/recipe probe и OOF, plan, experiment harness, flat package, integration commit/tree. Один чистый commit и один package используются обеими arms. Commit SHA своего содержимого в этот план не вписывается: после коммита SHA/tree и hash плана записываются в отдельный ignored manifest **до первого fit**. Documentation-only evidence commit разрешён после пары.

## Данные и split

TRAIN 87 670, TEST 41 166 строк. TEST labels отсутствуют; sample submission не содержит правильных ответов. Не искать TEST labels. Исходные пять файлов проверены в `/workspace/scratch/df1d37ea75a8/alfa-automl-agent/data/classification`; одинаковые свежие копии и повторная проверка hashes обязательны до/после каждой arm:

| Путь относительно task root | SHA-256 |
| --- | --- |
| `task.md` | `eebb55ebb8deb8257a2c012330ecc015e8d3a0bc3619c40da6c2a6ab200709bf` |
| `multiclass/participants/train.csv` | `171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0` |
| `multiclass/participants/test.csv` | `b825449603931b5f05a8855e3d92b57775221c93ed4730ac6b4c75b9ea608862` |
| `multiclass/participants/sample_submission.csv` | `4d23642fcff01e49b0ef5a9f018b2381c12fad97282fa586285ad49b01e9623e` |
| `multiclass/participants/data_dictionary.csv` | `13d2651c5fe02b7fc30f0e7290af45606aad5de76c2f7b69217abc270dbbcef4` |

Сохраняется фактическая штатная host split/group/seed policy базы. До fits manifest содержит её fingerprint, ordered row IDs/indices и raw feature-group hashes; после каждой arm проверяется равенство. ID/target исключены из групп и features; группы формируются до transformations. Внешние ответы остаются host-only; preflight использует только текущий разрешённый INNER-fit. Внутренний OOF — штатный grouped A/B seed 42 и три SGKF в A с seed 42. Random/grouped policy не подменяется ради покрытия.

## Eligibility, платный probe и очередь

Сначала `cheap_eligibility`: adaptive classification Macro F1 labels, разрешённая semantics, открытый INNER, measured baseline, совместимая grouped policy; 2–50 классов в точном порядке; fit ≤100 000, predict ≤500 000, features ≤512, каждая матрица ≤10 000 000 ячеек. Shape bounds проверяют INNER, outer и full-refit размеры без чтения outer labels. Numeric/categorical roles покрывают features без пересечения, ID/target исключены. До этих проверок нет subset exports/deep copies/augmentation. Затем `bounded_preflight` использует штатный `run_python` и cap **60 с** из текущего research allowance; staging, group hashing, A/B/fold support и проверки входят в те же 3 600 с. Моделей preflight не обучает и fit/trial slot не создаёт.

A/B минимум 100/60 строк, минимум 5 строк каждого класса, все OOF fit-fold содержат классы, group overlap 0. `coverage.available` не означает quality acceptance. Guard неизменен: grid .8/1/1.25, regularization .01, B gain ≥.01, class F1 drop ≤.02, максимум 2 раунда/128 кандидатов.

В ON после `recipes`/`seen`, до ordinary continuation, семантически ищется CB refinement iterations 400/depth10/learning_rate .03; номер `local_N` не используется. OFF ничего не переставляет. При фактическом запуске OOF найденный CB исключается из дальнейшей очереди; остальные recipes сохраняют относительный порядок. При skip до OOF исходная очередь сохраняется. После OOF-попытки слот не возвращается ради добавочного trial.

Разрешён **один cost-only identity probe только ON**, host cap **120 с**, с теми же текущими INNER inputs/base/dependencies. Текущий helper создаёт собственный compatible measurement; встреченный identity без совместимого cost receipt даёт skip, чужой timing не заимствуется. До probe, в том числе после preflight, должно оставаться 120 с research allowance сверх finalization reserve. Probe расходует trial slot и elapsed, но не участвует в quality selection, не обновляет best и не передаёт fitted-модели. Receipt, inputs и started/completed/cancelled events сохраняются даже при неуспехе.

После probe и подготовки заново измеряется доступное research-время. Для полного host wall `p`:

`estimated_oof = 4 * p * 1.5 + max(120, 2 * p)`

Только если `estimated_oof <= min(600, available_research_seconds)`, OOF допускается. Incomplete/timeout/incompatible probe или недостаток бюджета дают явный skip, уже потраченные время/slot сохраняются. Чужой baseline timing не заменяет probe. Это heuristic, не обещание длительности.

В действующем coordinator local cap 64. Реальные ранние invocations уменьшают оставшийся ON trial allowance; суммарный лимит не увеличивается. Probe не является бесплатным extension. Custom dedup использует `canonical_recipe_key(recipe, actual_code_hash)`; custom не отправляется в TPE `AdaptiveSearch.observe`.

## Полная стоимость и публикация

OOF invocation cap **600 с**, callback **90 с**, calibration заканчивается за **120 с** до исходного deadline, ещё 5 с оставлены на возврат/проверки. INNER получает `_assess_candidate(..., maximum=600)`; outer/full refit ограничиваются `stage_limit` как `min(600, allocator_timeout)`, не увеличивая исходный deadline. Children последовательны и используют действующий supervisor. OOF имеет до 10 физических fits на invocation; три достигнутые стадии INNER/outer/full TRAIN — до 30 fits, плюс 2 fits отдельного probe. При reject guard уже выполненные fits и копирование не обнуляются.

`V3State.measure_inner` и `record_timing` измеряют полный host `run_candidate`, а не последний `.fit()`. Эти exact source+recipe timings используются существующим `BudgetPolicy.finalization_estimate` для **candidate_outer и candidate_refit**, вместе с защищёнными baseline стадиями. На базе 3 600 с initial reserve floor 1 080 с; после собственных измерений всех pending recipes adaptive floor может снизиться до 72 с, но фактический reserve остаётся максимумом floor и полной оценки стадий. Publication buffer 2 с. Row scaling — оценка с safety factor 1.5, не гарантия. `research_budget.py` не меняется без доказанного отдельного integration request.

Custom проходит `_track_local_candidate`, `_assess_candidate`, host INNER, feasibility, terminal freeze, outer acceptance, fresh full TRAIN refit и обычную проверенную публикацию. После freeze новых trials нет. Неуспех custom не разрушает прежний baseline/checkpoint. Emergency target-prior CSV не квалифицирует OOF. Один frozen source/recipe/dependency snapshot во всех фазах; модели обучаются заново.

## Перед запуском: обязательный manifest и gates

Рабочие артефакты — ignored `data/round5-oof/`. Harness `experiment_round5_classification_native.py` имеет команды `prepare`, `smoke`, `run`, `audit`. `prepare` требует чистый committed checkout и новые evidence/workspace roots; сохраняет `run_manifest.json`, `manifest_seal.json`, `frozen_identity.json`, `frozen_oof.py`, `flat.zip` и host-only split/answers. Cap подготовки 600 с проверяется перед разрешением запуска; её elapsed учитывается отдельно. Каждая arm исполняется из одного flat package вне checkout, в независимых свежих task/output directories. Operational manifest содержит до fits:

- Чистый integration SHA/tree, hash плана, package и всех источников/dependencies; полные recipes/classes и подтверждение ожидаемого OOF hash.
- Пять input hashes и split/group fingerprints. Выбранный workspace — `/workspace/scratch/df1d37ea75a8/round5-native-runtime`; один `package/`, отдельные `off/task/task.md` и `on/task/task.md`; host reports — `data/round5-oof/off/` и `on/`.
- Абсолютный Python `/workspace/scratch/df1d37ea75a8/alfa-automl-agent/.venv/bin/python`, проверенная версия 3.12.14, фактические pinned package versions. Harness и solver запускаются тем же exact executable; ровно одна среда, без install/fetch/rebase посреди пары.
- CPU 8 threads, `CUDA_VISIBLE_DEVICES=""`, API credentials удалены штатным sanitized environment. Общий `/tmp/alfa-research-lab-cpu-0.lock` удерживается на всей последовательности OFF → audit → ON; дополнительно evidence coordinator lock.
- Наблюдаемый Linux cgroup: `cpu.max = 800000 100000`, `memory.max = 8589934592` — 8 CPU / 8 GiB; prepare повторно фиксирует среду. Одинаковый requested sampled tree threshold **6 GiB** и прежний `research_lab.process.supervise`. Native watchdog **3 630 с** при solver budget 3 600 с; 30 с обозначены как startup/cleanup allowance, solver не получает добавочного времени. Trusted cleanup имеет discovery 1 с, drain 2 с и root wait 5 с; фактические cleanup elapsed/status сохраняются. Timeout, memory stop или `cleanup.complete=false` останавливают серию. RSS/CPU/cleanup без телеметрии — unknown, successful publication тогда resource-provisional; child self RSS не является tree RSS.
- Реальные CLI `--classification-round4-mode {off,on}` и общие `--mode offline --research-search adaptive --search-profile legacy --time-budget 3600`; точные argv сохраняются в arm `started.json`.

До native проходят focused tests: default/explicit OFF sequence equality с прежним main; regression rescue; canonical identity/dedup/trial count/CB removal; early eligibility; paid probe и skip; reserve/absolute deadline/`maximum` forwarding; обычный lifecycle/fallback; class/dependency/input/receipt drift. Отдельный небольшой настоящий smoke из flat ZIP вне checkout проверяет импорты/fresh fit/prediction contract: synthetic fit 400 / predict 12 строк, worker cap 120 с и supervisor cap 150 с. Его actual synthetic fits считаются отдельно; он не доказывает применение OOF rule или native quality admission. Smoke и pair single-use; `run` требует successful smoke и не возобновляет уже начатую пару автоматически.

## Стопы и итоговые evidence

- OFF меняет прежнюю очередь/вызовы — исправить до native. Неподтверждённая frozen identity или незаполненный manifest — native не начинать.
- OFF повреждён, не завершён или не прошёл аудит — ON не запускать.
- Source/input/receipt drift, timeout или memory stop — сохранить evidence и остановить серию; автоматического повторного большого fit нет.
- OOF skip, guard rejection, INNER loss или feasibility/outer rejection — сохранить исход и причины; не ослаблять caps/guard, не менять очередь по результату.
- Отрицательный/неопределённый результат — завершить гипотезу с default OFF. При нехватке времени сохранить branch/source/evidence/checkpoint и точный статус.

Итоговые MD/JSON/CSV показывают requested/reached mode, INNER/published outer, actual winner и source/recipe hashes, все class metrics, ordinary/custom/probe attempts и физические fits, полные elapsed, skips/fallback/feasibility, frozen receipts всех достигнутых фаз, fresh full TRAIN evidence и TEST CSV hash. Независимый rescore и архив ограниченных raw evidence исключают модели и исходные большие datasets. Никаких API/Qwen, платформенной отправки, merge или выпуска принятого релиза.
