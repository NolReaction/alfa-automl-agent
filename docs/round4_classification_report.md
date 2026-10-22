# Раунд 4: классификация, измерения и ограничения

**OOF-кандидат прошёл заданные критерии качества на трёх confirmation development-срезах; full-TRAIN refit и host audit завершены.** Средний Macro F1 вырос с **0.6151288609 до 0.6302838409**, paired Δ **+0.0151549800**. На двух срезах guard применил правило, на одном сохранил identity; вклад Worms в прирост равен нулю. Цена — в среднем **204.874998 с против 21.360580 с** у фиксированного контроля. Результат передаётся для интеграционной проверки, статус **`integration_pending`**, default **OFF**. Полный native scheduler OFF/ON и платформенная отправка в этом задании не выполняются.

## База и границы сравнения

Исходная база: **`e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`**, ветка `work/round4-classification-v1`; реализация сохранена коммитом **`9ec3462d661d9bd8e8ff9b824890dd1286a3b374`**. План записан до обучения в `docs/round4_classification_plan.md`; машинный manifest **`data/r4c/plan.json`**, SHA-256 **`c639aa00c36d02d750dfdc11642c62b710ac8db88afec1465e5a5bed53d5ec51`**. После `prepare` runtime, стенд, recipes и план заморожены; новые метрики не используются для изменения вариантов.

Контроль `native_fixed` — обычный `BASELINE_SOURCE` текущего main, вызываемый изолированным v3 worker с исторически выбранным сильным рецептом PR #40: равное среднее реальных вероятностей LGB 400 (`num_leaves=15`, `reg_lambda=3`, `class_weight="balanced"`) и RF 120 (`depth=18`, `leaf=2`, `max_features="sqrt"`, `balanced_subsample`). Полный recipe сохранён в manifest; его hash `2f713e0d8c4d76b53f6fdb180a08ae99e20489ce84f7a074532c28a366511e02`, source hash `e06dbce5990fd68cfff61da62a46a1d0883da66897e38460ac6e9546ba1605cb`.

Это **новое обучение фиксированного рецепта на runtime e62**, а не повтор полного выбора scheduler. Стенд перед `prepare` сверил bytes соответствующих общих runtime-файлов с e62. Модельный код/классификационная очередь совпадают с d055, но общий host менялся; старые native-оценки не подставляются вместо новых. Отрицательные CB/LGB/XGBoost результаты PR #40 сохраняются историей и не объявляются результатами этого раунда.

## Проверенные данные и покрытие до моделей

TRAIN — **87 670 строк, 42 признака, 10 классов**; TEST — **41 166 строк без ответов**. Числовых признаков 39, категориальных 3. В признаках нет обнаруженных пустых/NULL/нечисловых числовых значений; отсутствующие значения не мотивируют отдельный поиск индикаторов. Есть **54 228 raw feature-групп**, **33 442** повторные строки сверх первых экземпляров; **14 829 строк** находятся в **1 458** группах с противоречивыми метками. Дубли `is_ftp_login`/`ct_ftp_cmd` сохранены. Это объясняет необходимость группировки, но не задаёт вычисленный потолок F1.

ID/target исключены из групп и признаков. Внешние split-индексы и ответы запечатаны до моделей; ответы хранятся только у host. A/B и три OOF-fold создаются внутри текущего fit, до feature transformations. На всех границах group overlap 0. Внешние seeds меняются по плану, внутренний seed и модельный seed остаются 42.

| Внешний seed | Fit / dev | A / B | Worms: fit rows/groups | Worms: A rows/groups | Worms: B rows/groups | Минимум класса в B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 70 155 / 17 515 | 55 945 / 14 210 | 49 / 49 | 41 / 41 | 8 / 8 | 8 |
| 43 | 70 283 / 17 387 | 56 026 / 14 257 | 53 / 53 | 46 / 46 | 7 / 7 | 7 |
| 44 | 70 481 / 17 189 | 56 298 / 14 183 | 49 / 49 | 39 / 39 | 10 / 10 | 10 |
| 137 | 69 884 / 17 786 | 56 303 / 13 581 | 57 / 57 | 46 / 46 | 11 / 11 | 11 |

Все OOF fit-fold содержат все классы. Для seed 42 минимальный класс представлен в трёх fold-fit/held как **27/14, 30/11, 25/16** строк; эти строки принадлежат разным группам. Preflight не блокирует запланированную OOF-попытку из-за известного дефицита B. Полная карта строк и независимых групп каждого класса во всех частях — `data/r4c/coverage_preflight.json`; её hash входит в manifest. Структурное `coverage.available` не заменяет последующую проверку guard.

Seeds 42/43/44 исследовались ранее, seed 137 также принадлежит той же публичной таблице. Все четыре среза остаются development evidence; никакой из них не становится независимым hidden TEST.

## Что изменено в кандидате

Новый probability adapter использует публичный `build_pipeline`, настоящий `predict_proba` и явное выравнивание class order; метрика задачи остаётся Macro F1 labels. Контроль identity должен дословно воспроизвести labels штатного источника. Внешний v3 wrapper возвращает labels, а probability matrices используются только внутри OOF и диагностических артефактов.

OOF: три fold внутри A создают OOF_A; на ней выбирается одно ограниченное правило. Отдельное обучение A → B проверяет его; B не меняет множители. Затем свежая модель обучается на всём разрешённом fit. `DEFAULT_GUARD_OPTIONS` не изменены: grid .8/1/1.25, regularization .01, gain в B ≥ 0.01, падение F1 класса ≤ 0.02; минимум 5 строк каждого класса в A/B. Добавлен ранний B-support preflight, экономящий ненужные fits при неизбежной identity. Для incomplete/late OOF — identity или отказ, а не частично настроенное правило.

Признаковые гипотезы ограничены двумя семействами: fit-only `log1p` частоты трёх категориальных признаков и три числовые направленные асимметрии `(forward−reverse)/(forward+reverse)`. Пары подтверждены словарём: объём в байтах, число пакетов, нагрузка bit/s. Имена публичных столбцов находятся только в recipe. При двух нулях — 0; отрицательные/nonfinite → NaN для штатной missing policy; масштабирование предотвращает overflow.

Отдельная гипотеза весов использует размер raw feature-группы только внутри fitting subset: `max(.25, n_group**−.5) / mean_fit_weight`. Она не использует конфликтность меток для назначения веса, не меняет исходные class weights и не читает prediction rows. Альтернативная модель — ровно один ExtraTrees 300/depth 18/leaf 4/sqrt/balanced на representation асимметрий. Новые комбинации после просмотра результатов не добавляются.

## Screening seed 42

Порядок и максимум fits заданы заранее: reference 2, identity 2, OOF 10, frequency 2, asymmetry 2, group weights 2, ExtraTrees 1 — **21 плановый fit**, hard cap **24** с учётом неуспехов. Автоматических retries нет. Время ниже включает весь host invocation, подготовку и проверки.

| Вариант | Состояние | Macro F1 | Δ к fixed control | Время, с | Fits started / completed | OOF правило trained / accepted / applied |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `native_fixed` | completed | 0.6270856863 | — | 22.598732 | 2 / 2, выведено из успешного рецепта | неприменимо |
| `identity` | completed | 0.6270856863 | 0.0000000000 | 24.372400 | 2 / 2 | 0 / 0 / 0 |
| `oof` | completed | 0.6458808663 | +0.0187951801 | 206.978815 | 10 / 10 | 1 / 1 / 1 |
| `frequency` | completed | 0.6319400355 | +0.0048543492 | 63.546723 | 2 / 2 | 0 / 0 / 0 |
| `asymmetry` | completed | 0.6287210542 | +0.0016353679 | 24.211353 | 2 / 2 | 0 / 0 / 0 |
| `group_weight` | completed | 0.6181931348 | −0.0088925514 | 23.304309 | 2 / 2 | 0 / 0 / 0 |
| `extra_trees_asymmetry` | completed | 0.4900924911 | −0.1369931951 | 8.707528 | 1 / 1 | 0 / 0 / 0 |

Всего **21/21 started/completed**, budget charge 21 при cap 24; retries, skips и process timeouts не было. Сумма семи invocation wall times — 373.719858 с; она не включает отдельную подготовку manifest/coverage. Все отрицательные исходы сохранены.

Identity подтвердил **точное совпадение всех ID/labels** с текущим fixed control. Измеренный probe 24.372400 с даёт допуск OOF: `4 * probe * 1.5 + max(120, 2 * probe) = 266.234399 с <= 600 с`. Это предварительная оценка, фактические callback deadlines остаются обязательными. Self peak RSS child identity — 563 019 776 байт; tree RSS этим не измерен.

OOF выбрал множители **Analysis 0.8, DoS 1.25, остальные 1** за 40 проверенных правил и 2 раунда. На OOF_A F1 изменился с 0.6004860470 до 0.6131690397; на отдельном B — с 0.6207966606 до 0.6405385691 (gain **0.0197419084**). Guard принял зафиксированное правило, новый full-fit применил его к dev-предсказаниям. Это реальные 10 физических обучений: восемь внутри четырёх callbacks и две финальные компоненты.

Внутри OOF времени на model callbacks ушло **59.949001 с**, на копирование subsets — **112.082435 с**, на guard — **0.148913 с**; общий OOF этап 172.396259 с, полный invocation 206.978815 с. Копирование составляет существенную часть стоимости; заявлять преимущество по одной длительности `.fit()` нельзя. Замороженная серия не оптимизируется после просмотра этого результата. Максимальный self RSS среди child runs OOF — 569 278 464 байт; это не tree RSS.

Изменения seed 42 по всем классам (TP/FP/FN — counts; вклад равен Δ F1/10):

| Класс | Support | Control TP/FP/FN | OOF TP/FP/FN | Δ F1 | Вклад в Macro Δ |
| --- | ---: | --- | --- | ---: | ---: |
| Analysis | 192 | 86 / 737 / 106 | 53 / 173 / 139 | +0.084130 | +0.008413 |
| Backdoor | 192 | 75 / 824 / 117 | 56 / 429 / 136 | +0.027947 | +0.002795 |
| DoS | 1 212 | 521 / 867 / 691 | 874 / 1 561 / 338 | +0.078529 | +0.007853 |
| Exploits | 3 296 | 1 772 / 255 / 1 524 | 1 753 / 233 / 1 543 | −0.002026 | −0.000203 |
| Fuzzers | 1 803 | 1 566 / 900 / 237 | 1 565 / 900 / 238 | −0.000297 | −0.000030 |
| Generic | 4 150 | 4 064 / 7 / 86 | 4 064 / 7 / 86 | 0 | 0 |
| Normal | 5 665 | 4 727 / 51 / 938 | 4 732 / 53 / 933 | +0.000351 | +0.000035 |
| Reconnaissance | 881 | 706 / 163 / 175 | 705 / 163 / 176 | −0.000682 | −0.000068 |
| Shellcode | 108 | 93 / 85 / 15 | 93 / 85 / 15 | 0 | 0 |
| Worms | 16 | 13 / 3 / 3 | 13 / 3 / 3 | 0 | 0 |

Основной эффект OOF на этом срезе — меньше ложных Analysis/Backdoor и больше найденных DoS при увеличении их FP. Worms не внёс прироста. Screen служил только выбору финалиста; отдельные confirmation-результаты приведены ниже.

Другие гипотезы дали более слабые или отрицательные результаты:

- Frequency: Δ +0.004854, из них **+0.003598** — вклад Worms; остальные девять классов вместе +0.001256. Этот одиночный результат не подтверждает устойчивую пользу признаков; recipe уступил OOF по правилу выбора и дальше не подбирался.
- Asymmetry: Δ +0.001635, вклад Worms снова +0.003598, остальных классов **−0.001963**. В частности, Δ F1 DoS −0.018536. Улучшение общей метрики здесь объясняется редким классом.
- Group weights: Δ −0.008893, Δ F1 DoS −0.048663 и Worms −0.038306; вклад остальных классов тоже отрицателен (−0.005062). Ограниченное снижение веса дублей не улучшило этот fixed base.
- ExtraTrees на асимметриях: Δ −0.136993. Worms получил 15 TP, но 217 FP (контроль 13 TP/3 FP), F1 упал с 0.812500 до 0.120968; Shellcode сохранил 93 TP, но FP выросли с 85 до 433. Более регуляризованный заданный рецепт не устранил чрезмерные rare-class срабатывания. Это отрицательный результат конкретного frozen варианта, не всей семьи ExtraTrees.

## Confirmation и классовый разбор

По правилу максимального screen Macro F1 выбран **один** финалист `oof`; правило у него применено. Выбор запечатан до первого confirmation в `data/r4c/confirmation_selection.json`, SHA-256 **`0072c2ab32725ebfdf146340f7b2f785a4482b1e3ac35c487d718fc95ae1b0b1`**. Source hash — `1cb8a3f9df7588b5c4773351b9426c29eedd4c55f297f59fafeeaeefaf300862`, recipe hash — `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc`; dependency hashes зафиксированы в исходном manifest. Запланированы только 43/44/137, максимальная цена **36 fits**. Параметры и выбор других гипотез по confirmation не меняются.

| Seed | Fixed control F1 | Финалист F1 | Paired Δ | Вклад всех классов кроме редчайшего | Состояние |
| --- | ---: | ---: | ---: | ---: | --- |
| 43 | 0.6077597140 | 0.6340572713 | +0.0262975573 | +0.0262975573 | completed; правило принято |
| 44 | 0.6189618195 | 0.6189618195 | 0.0000000000 | 0.0000000000 | completed; identity |
| 137 | 0.6186650493 | 0.6378324319 | +0.0191673826 | +0.0191673826 | completed; правило принято |
| Среднее confirmation | **0.6151288609** | **0.6302838409** | **+0.0151549800** | **+0.0151549800** | критерии пройдены |

Причины OOF-решения и фактическая стоимость; gain в B не является внешней confirmation-оценкой:

| Seed | Время control / OOF, с | Fits OOF | Trained / accepted / applied | B Macro F1 gain | Решение |
| --- | ---: | ---: | --- | ---: | --- |
| 42, screen | 22.598732 / 206.978815 | 10 | 1 / 1 / 1 | +0.0197419084 | accepted |
| 43 | 21.284204 / 206.807524 | 10 | 1 / 1 / 1 | +0.0186453471 | accepted |
| 44 | 21.206085 / 203.768150 | 10 | 1 / 0 / 0 | +0.0170861230 | selection_class_drop → identity |
| 137 | 21.591451 / 204.049321 | 10 | 1 / 1 / 1 | +0.0169921298 | accepted |

На seed 43 применены множители Analysis 0.8, Backdoor 0.8, DoS 1.25, остальные 1. Внешний прирост распределён главным образом между Analysis, Backdoor и DoS; вклад Worms равен нулю. На seed 44 выбранное внутри A правило имело множители Analysis 0.8, DoS 1.25, Exploits 0.8, Fuzzers 0.8, остальные 1. Хотя общий B gain положителен, Δ F1 Shellcode в B **−0.0209634255** нарушил неизменный внутренний порог −0.02. Guard вернул все множители 1; внешний результат и все class counts совпали с контролем. Это **identity при полной цене OOF**, а не улучшенный срез. Guard после результата не перенастраивался.

Основной критерий — средний paired-прирост **≥ 0.003 только по confirmation**, при Δ каждого среза ≥ −0.01 и Δ F1 каждого класса ≥ −0.05. Дополнительно средний суммарный вклад остальных классов после исключения глобально редчайшего (Worms, 65 TRAIN строк) должен быть положительным. Основная Macro F1 всегда включает все 10 классов. Исключение редчайшего используется лишь как отдельная проверка устойчивости.

На seed 137 принято правило Analysis 0.8, DoS 1.25, остальные 1. Всего на confirmation обучены **3** правила, приняты и применены **2 из 3**. Все шесть invocation завершены; израсходованы **36/36 fits** (6 у контроля и 30 у OOF), без retries. Сумма invocation wall times — 678.706735 с. OOF требует примерно **9.6×** времени фиксированного контроля на этой машине; reject в B не возвращает уже затраченную стоимость.

Все четыре заранее заданных условия пройдены: средний Δ **+0.0151549800 ≥ +0.003**, худший Δ среза **0 ≥ −0.01**, худший Δ F1 класса **−0.0057558743 ≥ −0.05** (Exploits, seed 137), вклад без редчайшего класса **+0.0151549800 > 0**. Primary mean использует только 43/44/137; screen 42 в него не включён.

Вклад всех классов в средний confirmation-прирост; последний столбец — среднее трёх `ΔF1_class / 10`:

| Класс | Δ F1 seed 43 | Δ F1 seed 44 | Δ F1 seed 137 | Средний вклад в Macro Δ |
| --- | ---: | ---: | ---: | ---: |
| Analysis | +0.119404 | 0 | +0.097670 | +0.00723577 |
| Backdoor | +0.041011 | 0 | +0.021769 | +0.00209266 |
| DoS | +0.106346 | 0 | +0.075626 | +0.00606574 |
| Exploits | −0.004171 | 0 | −0.005756 | −0.00033091 |
| Fuzzers | +0.000172 | 0 | +0.000051 | +0.00000743 |
| Generic | 0 | 0 | 0 | 0 |
| Normal | +0.000214 | 0 | +0.000247 | +0.00001536 |
| Reconnaissance | 0 | 0 | −0.000183 | −0.00000608 |
| Shellcode | 0 | 0 | +0.002250 | +0.00007501 |
| Worms | 0 | 0 | 0 | 0 |

Confusion показывает конкретный обмен ошибок. На seed 43 число Exploits → Analysis сократилось **467 → 35**, DoS → Analysis **345 → 14**, Exploits → Backdoor **306 → 77**, но Exploits → DoS выросло **602 → 1 290**. На seed 137 те же изменения: Exploits → Analysis **433 → 48**, DoS → Analysis **290 → 28**, Exploits → Backdoor **339 → 262**, Exploits → DoS **609 → 1 108**. Снижение FP редких Analysis/Backdoor и рост найденных DoS дали положительный macro effect, при небольшом ухудшении Exploits. На seed 44 confusion не изменилась. Полные TP/FP/FN/support и матрицы 10×10 каждого опыта сохранены в `record.json`; срезы не объединяются в фиктивный независимый тест.

Решение по качеству — передать frozen OOF-кандидат для последующей интеграционной проверки. Это подтверждение на запланированных development-срезах, не оценка hidden TEST и не гарантия прироста платформенных баллов. Общий host не изменён, native OFF/ON на общем integration SHA остаётся невыполненным, default остаётся OFF.

## Delivery

Frozen `oof` заново обучен на **всех 87 670 TRAIN строках** и создал **41 166 unscored TEST predictions** за **255.655323 с**, с **10/10 started/completed fits**. Source и recipe hashes совпадают с confirmation selection; evidence содержит `fresh_full_refit=true`, `full_fit_rows=87670`, `predict_rows=41166`. Внутреннее обучение/проверка правила заново использовали только TRAIN: A 70 155, B 17 515 строк; затем обе финальные компоненты получили весь TRAIN. Правило обучено, принято и применено — **1/1/1**; множители Analysis 0.8, DoS 1.25, остальные 1.

Host проверил ID/order, число строк, допустимые labels, receipts и соответствие опубликованного CSV worker output. Результат — `data/r4c/trials/oof_full_refit/final_predictions.csv`; SHA-256 **`c57583819d69e7cc59eca1859830cc1d3b2d4796160e09127a114cd11521bdbf`**. Полная запись: `data/r4c/trials/oof_full_refit/record.json`. Максимальный child self RSS — **674 807 808 байт**, отдельно от неизвестного tree RSS. TEST не имеет доступных ответов: качество не оценивалось, платформенной отправки не было.

Вся публичная серия завершила **67 физических fits**: 21 screen + 36 confirmation + 10 delivery; все **14 invocation** успешны. Full refit подтверждает доставку выбранного кандидата и не добавляет ещё одну оценку качества.

## Ресурсы, учёт и проверка

Python **3.12.14**; используемые sklearn 1.5.2, LightGBM 4.6.0, NumPy 1.26.4, pandas 2.2.3 соответствуют официальным pins. Полный перечень установленных/отсутствующих пакетов сохранён в manifest. Наличие отдельных официальных пакетов не заявляется для всей среды; новые runtime dependencies не добавлены.

Обычный trial cap 300 с, OOF 600 с, callback 90 с, final reserve 120 с. Nested callbacks исполняются существующим `run_python` последовательно и ограничены настоящим процессным timeout и исходным абсолютным deadline. Фактический общий cgroup — **8 CPU / 8 GiB**. Tree RSS и cleanup при недоступном process monitor — **unknown/null**; запрошенный 6 GiB sampled tree cap не считается применённым. Child `getrusage(RUSAGE_SELF)` хранится отдельно как **self RSS**, не как пик всего дерева. Конкурсные 32 GB/10 500 с не квалифицированы.

Physical fits у новых вариантов подтверждаются child `started/completed` events. `cancelled_before_fit` вычитается из фактически начатых, но учитывается консервативно в budget charge. При убийстве родителя и неполных доказательствах actual totals остаются null, а budget расход резервируется. У успешного `native_fixed` нет fit events: 2 fits выводятся из двухкомпонентного source и успешного host receipt, с явным provenance; это не имитация измеренного event counter.

API attempts/responses 0/0 имеют основание `structural_no_provider_path`: стенд/кандидат не создают API-клиент. Это не счётчики полного native scheduler. Прошли **181 различный целевой тест**: 171 model/contract test и 10 stdlib export-integrity tests. Первые 171 включают 52 новых и 119 существующих тестов, проверенных в двух успешных invocations: консолидированные 170 за 6.276 с, затем 17 harness-тестов (16 повторных и 1 дополнительный). Включены настоящий маленький probability adapter, v3 identity protocol на синтетике и accounting timeout-edge; публичные TRAIN-модели эти тесты не обучали. Exporter и его тесты добавлены отдельно от frozen runtime/harness/plan.

**Финальный host audit: 14/14 completed invocation проверены**, все 14 исходов сохранены. `data/r4c/audit.json` подтверждает source/input/split/recipe/output hashes, receipts nonce/phase, fit accounting, пересчёт сохранённых dev CSV, точное identity label agreement и неизменность официальных входов. Confirmation status — `confirmed`, recommendation — `integration_pending`. Отдельная проверка извлечённого финального ZIP выполняется после коммита документов; её точные counts и final head относятся к сопроводительному handoff, а не заявлены здесь заранее.

## Воспроизведение

Изменены **15 файлов** в разрешённых границах; точные final head SHA и PR указаны в сопроводительном handoff после коммита отчёта:

- Runtime (4): `agent/oof_classification.py`, `agent/round4_classification_candidate.py`, `agent/round4_classification_probability.py`, `agent/round4_classification_features.py`.
- Стенды (2): `experiment_round4_classification.py`, `experiment_round4_classification_export.py`.
- Тесты (6): `tests/test_round4_classification_candidate.py`, `tests/test_round4_classification_experiment.py`, `tests/test_round4_classification_export.py`, `tests/test_round4_classification_features.py`, `tests/test_round4_classification_preflight.py`, `tests/test_round4_classification_probability.py`.
- Документы (3): `docs/round4_classification_plan.md`, `docs/round4_classification_report.md`, `docs/round4_classification_integration.md`.

Реализованные команды стенда из checkout с зафиксированными sources и подготовленной официальной средой. В Linux все пять стадий завершены с `data/r4c`; для повторной серии нужен другой новый короткий output root.

```bash
python experiment_round4_classification.py prepare --mode offline --output data/r4c
python experiment_round4_classification.py screen --mode offline --output data/r4c
python experiment_round4_classification.py confirm --mode offline --output data/r4c
python experiment_round4_classification.py full-refit --mode offline --output data/r4c
python experiment_round4_classification.py audit --mode offline --output data/r4c
```

PowerShell: тот же интерфейс, короткий новый путь `C:\r4c` для снижения риска MAX_PATH. **PowerShell/Windows execution в этой сессии не проверено**; команды ниже документируют воспроизведение, а не результаты Windows-теста. `$classificationPython` должен указывать на подготовленный Python 3.12.

```powershell
$classificationPython = '.\.venv\Scripts\python.exe'
& $classificationPython experiment_round4_classification.py prepare --mode offline --output C:\r4c
& $classificationPython experiment_round4_classification.py screen --mode offline --output C:\r4c
& $classificationPython experiment_round4_classification.py confirm --mode offline --output C:\r4c
& $classificationPython experiment_round4_classification.py full-refit --mode offline --output C:\r4c
& $classificationPython experiment_round4_classification.py audit --mode offline --output C:\r4c
```

Для `--task` можно явно указать штатный `data/classification/task.md`; default уже ведёт к нему.

Переносимый пакет **`round4-classification-evidence.zip`** создаётся после коммита финальных документов. В него входят answers development-срезов, ID-only TEST, predictions, manifests, receipts, child metadata/probability arrays, логи и source snapshot. Исходные feature-таблицы, сериализованные requests и модели исключены. Exporter проверяет request/probability hashes, disk/embedded metadata, receipts и точные worker-local fit/predict indices; сводка — `nested_child_audit.json`. Проверку bytes исключённых requests нельзя повторить из архива; привязку worker-local строк к исходному TRAIN/TEST проверяет host audit.

Команда упаковки из checkout, с новым абсолютным output path вне evidence root:

```bash
python experiment_round4_classification_export.py --input data/r4c --output "$PWD/round4-classification-evidence.zip"
```

```powershell
& $classificationPython experiment_round4_classification_export.py --input C:\r4c --output C:\round4-classification-evidence.zip
```

Независимая проверка скачанного архива требует только Python 3.10+ stdlib, без ML-пакетов и исходных таблиц. `rescore.py` сверяет `evidence_manifest.json`, SHA-256, схему, уникальные упорядоченные ID, классы и сохранённый development Macro F1 с допуском `1e-12`; TEST CSV проверяется только по формату. Linux, из каталога с ZIP:

```bash
python3 -m zipfile -e round4-classification-evidence.zip extracted
cd extracted/round4-classification-evidence
python3 rescore.py . > independent_rescore.json
```

PowerShell, документированный, но не проверенный в этой сессии вариант с новым коротким extraction root:

```powershell
Expand-Archive -LiteralPath .\round4-classification-evidence.zip -DestinationPath C:\r4e
Set-Location C:\r4e\round4-classification-evidence
py -3 .\rescore.py . | Set-Content -Encoding utf8 .\independent_rescore.json
```

Компактная машинная точка входа — `data/r4c/results_summary.json`: 14 исходов, paired confirmation, учёт 67 fits и host audit. Это производная сводка сохранённых результатов, без новых оценок или подбора. Подробные evidence: `data/r4c/plan.json`, `seal.json`, `coverage_preflight.json`, `confirmation_selection.json`, `experiments.json`, `audit.json` и `trials/*/record.json`; внутри worker — host receipt, saved predictions, `round4_evidence.json`, child metadata/events. План/источники не переписываются при дополнении этого отчёта. Точный контракт будущего подключения — `docs/round4_classification_integration.md`.
