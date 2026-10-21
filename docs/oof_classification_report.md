# OOF Macro F1: независимый допуск и перенос после refit

**Включение по умолчанию не обосновано.** На confirmation все 9 правил остались
identity. На development принято одно из 6 правил: оно ухудшило outer Macro F1
на 0.003465 и дало +0.000995 на отдельном production audit после полного refit.
Четыре дополнительных calibration fit потребовали около 4.6 s на пару.
Это отрицательный результат для автоматического включения; пороги не снижались.

## База, история и область изменений

База до опытов: `664bd5a38c4ba8e1451163917bbdff93979b4b54`, после merge #16.
Проверки предков `6e7b5c019a438c3e5187ef6c96107bb329cb5678` и
`7d54bd6d9179cf57e4cfc4b39d3762092fcdd0aa` вернули 0.
Ветка `work/oof-macro-f1-v1`; PR https://github.com/NolReaction/alfa-automl-agent/pull/20 .

Код заморожен до первого audit в локальном commit
`f773b9bdaad98259eb00edf0f34c4e8d6ad17ef5`, clean checkout. Опубликованный
эквивалент — `1939016fd009ca995fc3ee442401bf451317b8d1`. Дерево обоих
побайтно одинаково: `2b857d1e80332a579c4233eb7a109b705a548f29`.
Локальный commit сохранён в Git bundle; различаются только metadata коммитов.
Позднейшее оформление отчёта не меняет runtime-исходники замороженного плана.

Изменены только шесть согласованных файлов: чистый OOF-модуль, его тесты,
генератор, экспериментальный host, протокол и этот отчёт. Existing decision,
Research v3, общий runtime/model search, splits, CLI, budget, приложение,
requirements и ZIP не менялись. Готовой интеграции с Research v3 здесь нет.

Предыдущие decision-эксперименты изучены до реализации: старый pilot дал
3 wins/6 losses, guarded follow-up — 2 wins/7 identity. Их seeds и audit не
переименовываются в новое подтверждение. Нынешние семейства также известны;
новыми являются заранее заданные реализации и отдельная проверка refit.
Поиск новых seeds в прежних decision reports/generators не нашёл этих случаев.

## Зафиксированный протокол и модель

Полный протокол: [oof_classification_protocol.md](oof_classification_protocol.md).
Development: seeds 131/379, confirmation: 2203/4001/10009; три семейства.
Дополнительно два structural/support probe, seed 131. Все 17 источников,
схемы и разбиения подготовлены до первого обучения основной серии; после результатов нет
подбора параметров, resplit, изменения поддержки или сетки guard.

Synthetic имеет 1500 allowed, 400 outer и отдельные 400 production audit.
A/B — 1050/450 строк, у дублей 1048/452, группы из четырёх строк целые.
Три GroupKFold строятся только в A. Четвёртая модель учится на A и предсказывает
B. Ответы B не входят ни в один calibration fit или preprocessing. Кандидат
выбирается на OOF-A; B может только принять его или identity. Старые guard
минимумы 5/class в обеих частях, gain>=0.01 и max per-class drop<=0.02 сохранены.

Правило замораживается перед allowed→outer fit. Затем **то же** правило, без
retune, применяется после production-refit на allowed+outer=1900 строк к
отдельным 400 production audit. Это полный train относительно выделенного
development-набора; production audit никогда не обучает модель или правило.
Внутри OOF-плеча identity и правило используют одну модель и те же вероятности.

Один RandomForest: 64 trees, depth 10, min leaf 2, sqrt features,
balanced_subsample, 8 threads, seed случая. Numeric median и categorical
most_frequent+OneHot(ignore unknown) заново fit только на fitting-подвыборке.
Модель и preprocessing не подбираются по B/outer/production audit.
Categorical shift здесь IID/domain shift без времени; настоящая time policy
возвращает `unsupported_time_policy`/identity, без random OOF.

Оба плеча имеют один общий cap 180 s (public 900 s, если доступен), включая
calibration, selection, оба refit и prediction. Calibration использует исходный
arm deadline минус 60 s резерва, subprocess cap 25 s. Порядок плеч чередуется.
Все fit последовательны; тяжёлые suites одновременно не запускались.
Фиксированный контроль не использует остаток cap для AutoML search: это
парный equal-cap fixed-model контроль, **не доказательство выигрыша над AutoML**.

## Все реальные пары

Macro F1, больше лучше. Identity здесь — контроль на тех же вероятностях
финальной модели OOF-плеча. Независимое контрольное плечо дало те же метки и
F1 для каждой строки таблицы. Вероятности разных RF-процессов отличаются лишь
порядком суммирования до 3.33e-16: все 34 сравнения фаз allclose(atol=1e-12),
argmax совпадает. Побайтное равенство разных процессов не заявляется.

| Семейство / seed | Этап | Outer identity | Outer rule | Production identity | Production rule | Допуск |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `imbalance_overlap-131` | development | 0.81867967 | 0.81521456 | 0.78973322 | 0.79072826 | accepted |
| `imbalance_overlap-379` | development | 0.88395867 | 0.88395867 | 0.84899398 | 0.84899398 | selection gain |
| `contradictory_duplicates-131` | development | 0.45705048 | 0.45705048 | 0.44187004 | 0.44187004 | selection gain |
| `contradictory_duplicates-379` | development | 0.45988728 | 0.45988728 | 0.44130195 | 0.44130195 | selection gain |
| `categorical_shift-131` | development | 0.53209259 | 0.53209259 | 0.58544770 | 0.58544770 | selection gain |
| `categorical_shift-379` | development | 0.59744973 | 0.59744973 | 0.56391742 | 0.56391742 | identity tuning |
| `imbalance_overlap-2203` | confirmation | 0.77606898 | 0.77606898 | 0.74271109 | 0.74271109 | identity tuning |
| `imbalance_overlap-4001` | confirmation | 0.82794009 | 0.82794009 | 0.80273182 | 0.80273182 | identity tuning |
| `imbalance_overlap-10009` | confirmation | 0.76689380 | 0.76689380 | 0.81305342 | 0.81305342 | selection gain |
| `contradictory_duplicates-2203` | confirmation | 0.47390462 | 0.47390462 | 0.43739099 | 0.43739099 | identity tuning |
| `contradictory_duplicates-4001` | confirmation | 0.49856756 | 0.49856756 | 0.51268745 | 0.51268745 | identity tuning |
| `contradictory_duplicates-10009` | confirmation | 0.50138897 | 0.50138897 | 0.46858416 | 0.46858416 | selection gain |
| `categorical_shift-2203` | confirmation | 0.54699353 | 0.54699353 | 0.58972994 | 0.58972994 | identity tuning |
| `categorical_shift-4001` | confirmation | 0.55155456 | 0.55155456 | 0.50622654 | 0.50622654 | identity tuning |
| `categorical_shift-10009` | confirmation | 0.58182462 | 0.58182462 | 0.58665650 | 0.58665650 | selection gain |
| `rare_support-131` | degenerate | 0.61218233 | 0.61218233 | 0.61228033 | 0.61228033 | rare support |
| `impossible_split-131` | degenerate | 0.15969582 | 0.15969582 | 0.17192331 | 0.17192331 | split unavailable |

| Этап | Пары | Принято правил | Outer win / tie / loss | Production win / tie / loss |
| --- | ---: | ---: | --- | --- |
| Development | 6 | 1 | 0 / 5 / 1 | 1 / 5 / 0 |
| Confirmation | 9 | 0 | 0 / 9 / 0 | 0 / 9 / 0 |
| Support/split probes | 2 | 0 | 0 / 2 / 0 | 0 / 2 / 0 |

Все 17 пар завершены, skipped/failed arms нет. Identity — отказ от изменения,
а не полезная калибровка. На confirmation 6 identity_on_tuning и 3 отказа
selection_gain_below_minimum. На development — 1 accepted, 1 identity_on_tuning,
4 отказа по selection gain. Средний «score платформы» по семействам не считается.

### Единственное принятое правило и все его ухудшения

`imbalance_overlap/131`: множители `[0.8, 1, 1.25]` в порядке `01`, `NA`, `class c`.
A support `[763,221,66]`, B support `[327,89,34]`. B Macro F1 вырос на 0.02570489;
это только внутренний допуск, не гарантия внешнего роста.
Outer Δ=-0.00346511; production после refit Δ=+0.00099504.
Даже при небольшом production Macro F1 росте F1 редкого класса снизилась.

| Фаза | Класс | Support | F1 identity | F1 rule | Разница |
| --- | --- | ---: | ---: | ---: | ---: |
| outer | `01` | 289 | 0.95918367 | 0.95711835 | -0.00206532 |
| outer | `NA` | 77 | 0.85534591 | 0.85889571 | +0.00354979 |
| outer | `class c` | 34 | 0.64150943 | 0.62962963 | -0.01187980 |
| production | `01` | 283 | 0.94077834 | 0.93771626 | -0.00306208 |
| production | `NA` | 87 | 0.79012346 | 0.80701754 | +0.01689409 |
| production | `class c` | 30 | 0.63829787 | 0.62745098 | -0.01084689 |

Полные per-class precision/recall/F1/support **каждого** случая и обоих плеч
сохранены в `report.json` и `cases/<case>/<arm>/result.json`. Там же все
reasons, candidate_rule до допуска, выбранное rule, fit counts и timings.
Ни одно ухудшение не отфильтровано из таблицы.

`rare_support`: 6 редких allowed labels, A/B=4/2; guard даёт
`tuning_insufficient_class_support`, calibration fit=0. `impossible_split`:
одна allowed raw-группа, host возвращает явный
`host_insufficient_groups_for_tuning_selection`, calibration fit=0. В обоих
случаях identity-модель и production-refit всё же исполнены, поэтому CSV
и метрики существуют; корректность OOF для невозможного split не заявляется.

## Реальная цена

15 основных пар: OOF 3+1 calibration, затем outer model и production-refit:
6 fit против 2 у контроля. Дополнительно ровно четыре fit. Два probes:
по 2 fit в каждом плече, без calibration. Всего в основной серии **128 настоящих
fit**: 60 calibration и 68 outer/production. Fit из unit execution proof
учитывается отдельно, не как дополнительная исследовательская пара.

| Основные 15 пар, wall time | Мин., s | Медиана, s | Макс., s |
| --- | ---: | ---: | ---: |
| Identity, включая production | 2.176 | 2.341 | 2.655 |
| OOF, включая production | 6.657 | 7.024 | 7.684 |
| Только четыре calibration fit | 4.286 | 4.596 | 5.264 |

Контроллер всей серии: **154.092 s**, включая запуск дочерних
процессов и сбор отчёта. Точная дополнительная calibration wall сумма: 69.382 s.

| Пара | A / B support (`01`, `NA`, `class c`) | Calibration fits | Control / OOF fits | Control / OOF seconds |
| --- | --- | ---: | --- | --- |
| `imbalance_overlap-131` | [763, 221, 66] / [327, 89, 34] | 4 | 2 / 6 | 2.644 / 7.097 |
| `imbalance_overlap-379` | [758, 215, 77] / [327, 87, 36] | 4 | 2 / 6 | 2.183 / 7.023 |
| `contradictory_duplicates-131` | [405, 369, 274] / [183, 148, 121] | 4 | 2 / 6 | 2.176 / 6.679 |
| `contradictory_duplicates-379` | [402, 369, 277] / [186, 142, 124] | 4 | 2 / 6 | 2.232 / 6.691 |
| `categorical_shift-131` | [419, 325, 306] / [164, 151, 135] | 4 | 2 / 6 | 2.189 / 6.867 |
| `categorical_shift-379` | [392, 321, 337] / [169, 155, 126] | 4 | 2 / 6 | 2.341 / 6.913 |
| `imbalance_overlap-2203` | [771, 198, 81] / [326, 86, 38] | 4 | 2 / 6 | 2.281 / 6.657 |
| `imbalance_overlap-4001` | [756, 214, 80] / [337, 84, 29] | 4 | 2 / 6 | 2.587 / 7.058 |
| `imbalance_overlap-10009` | [763, 210, 77] / [329, 90, 31] | 4 | 2 / 6 | 2.372 / 7.360 |
| `contradictory_duplicates-2203` | [399, 353, 296] / [165, 168, 119] | 4 | 2 / 6 | 2.376 / 7.024 |
| `contradictory_duplicates-4001` | [397, 355, 296] / [183, 133, 136] | 4 | 2 / 6 | 2.381 / 6.715 |
| `contradictory_duplicates-10009` | [399, 382, 267] / [185, 141, 126] | 4 | 2 / 6 | 2.655 / 7.409 |
| `categorical_shift-2203` | [416, 324, 310] / [183, 137, 130] | 4 | 2 / 6 | 2.649 / 7.243 |
| `categorical_shift-4001` | [405, 339, 306] / [167, 137, 146] | 4 | 2 / 6 | 2.241 / 7.684 |
| `categorical_shift-10009` | [387, 324, 339] / [165, 171, 114] | 4 | 2 / 6 | 2.315 / 7.371 |
| `rare_support-131` | [530, 516, 4] / [211, 237, 2] | 0 | 2 / 2 | 2.260 / 2.452 |
| `impossible_split-131` | [0, 0, 0] / [0, 0, 0] | 0 | 2 / 2 | 2.225 / 2.456 |

Настроенный memory monitor cap — 8 GiB, но в данном PID namespace psutil
inventory недоступна: все 128 execution receipts честно имеют resources.available=false,
cleanup_complete=null. Подтверждено завершение/reaping root child; полная
инвентаризация потомков в этом окружении не сертифицируется. Worker использует
RF/joblib threading, не process backend. Это не adversarial sandbox.
Независимое измерение worker RSS через getrusage доступно: **152.73–189.43 MiB**,
включает жизнь worker, не supervisor. RSS не выдаётся за соблюдение жёсткого
8 GiB allocation cap. CPU time/process-tree RSS не выдумывались.

## Проверки и независимый audit

На использованных project pins: Python 3.12.14, NumPy 1.26.4, pandas 2.2.3,
SciPy 1.12.0, scikit-learn 1.5.2, joblib 1.4.2, threadpoolctl 3.6.0,
psutil 6.1.1. Установлен существующий requirements; зависимости проекта не менялись.
Это subset библиотек, не воспроизведение полного Docker организаторов.

```bash
python -m unittest tests.test_oof_classification tests.test_classification_decision -v
# Ran 89 tests in 2.490s
# OK
python -m py_compile agent/oof_classification.py oof_classification_cases.py experiment_oof_classification.py tests/test_oof_classification.py
git diff --check
```

46 новых tests + 43 существующих decision tests. Callback-spy проверяет
отсутствие B в fit, OOF coverage и группы; intervention B labels не меняет
candidate_rule. Проверяются literal labels/column permutation, malformed
indices/probabilities, missing/partial OOF, unsupported time, callback failures,
весь исходный deadline, расходы deep-copy до запуска и истечение при возврате.
Модуль не меняет входные таблицы (включая nested object cells) и правило;
guard options используются как замороженный снимок даже при изменении исходного
словаря внешним callback. Host tests проверяют actual TaskSpec API, freeze до обоих refit
и scoring, preregistration tamper и отказ re-audit на неполном отчёте.

Реальный небольшой дочерний fit подтвердил fit-only preprocessing: median=2
при predict=999; predict-only category `novel` отсутствует в fitted encoder.
Receipt exit=0, timed_out=false; request не содержит predict labels, модели
и вероятности сохранены с хешами. Это реальный execution proof, а не spy-fit.

Штатный re-audit: **34/34 arms verified, 0 новых fit**. Все 128 сохранённых
моделей повторно предсказали свои X_predict; вероятности совпали с atol=1e-12.
Проверены hashes source/input/splits/rule/request/model/probabilities, exit
receipts, полное покрытие плана, реальные fit rows/labels/classes/config,
JSON guard из восстановленных OOF и frozen rule после обоих refit.
Macro F1 и все per-class показатели независимо пересчитаны прямыми confusion
counts, без sklearn scorer. Audit-проверки не основаны на Python assert. Отдельный настоящий запуск с
`python -O` также дал 34/34 verified, 0 model fits.

Отдельный проверяющий до измерений подтвердил все 17 source/CSV/splits,
raw duplicate groups, A/B и held coverage, нулевое включение B в fit и budgets:
`independent_split_audit.json`.

Второй независимый проверяющий завершил `independent_results_audit.json`:
17 пар / 34 arms, 128 успешных fit receipts, 384 hashes модели/запроса/вероятностей,
136 наборов метрик и 408 per-class строк совпали. Он использовал собственные
stdlib CSV, argmax и TP/FP/FN-формулы, без host re-audit, sklearn scorer,
rule fit или model fit. Отдельно проверены fit-only medians/categories,
перестановка probability columns, все guard scores/пороги допуска и неизменность
множителей. Расхождений нет; B-fit rows=0 во всех 60 calibration fit.

Native Linux/Windows CI public code SHA `1939016f`:
https://github.com/NolReaction/alfa-automl-agent/actions/runs/39605755638 .
Linux job завершился успешно: 756 core tests за 444.252 s, 227 app tests
за 53.574 s, проверка Windows launcher и штатная сборка также прошли.
Windows job на момент оформления отчёта ещё выполняется. Локальный полный
dev_check/app suite в этой задаче не запускался; 89 профильных tests не называются
полным suite. CI относится к точному public code SHA выше; последующие правки
только этого Markdown-отчёта не меняют проверяемый Python.

## Ограничения и решение

Confirmation — свежие seeds заранее замороженных **известных семейств**, не
доказательство устойчивости на неизвестных задачах. После этой серии они
просмотрены; следующий подбор по ним будет exploratory. Публичный
`data/classification/task.md` и CSV отсутствуют: public folds 0/1/2 получили
`skipped_missing_data`, а не passed. Замена из сети не скачивалась.

Только одна небольшая фиксированная модель. Ни AutoML search за одинаковое
время, ни сильные модели, ни реальные platform hidden tasks здесь не сравнивались.
Не утверждается преимущество над AutoML, улучшение public score или причина
прежних нулевых platform scores. Порогов для редких классов не снижали.
Девять confirmation identity за четыре лишних fit не оправдывают автоматическое
включение. Рекомендуется оставить модуль как отдельный opt-in эксперимент;
дальнейшую гипотезу фиксировать до нового внешнего набора.

## Точные hooks для будущей интеграции владельцем runtime

1. Metric gate: вызывать только для Macro F1. AUC/LogLoss используют исходные
   вероятности; `predict_with_rule` возвращает только метки.
2. Host передаёт только outer-fit X/y и raw-duplicate group IDs, явную policy,
   заранее фиксированные local A/B indices и три A-only folds. Для времени
   использовать explicit unsupported/identity, без random fallback.
3. Нужен отдельный bounded probability callback. Текущий Macro F1 callback
   Research v3 возвращает метки; нельзя считать его совместимым с OOF API.
   Каждый fit получает только X_fit/y_fit/X_predict, заново fit preprocessing,
   возвращает `{'probabilities': p, 'classes': exact_column_labels}`. Timeout
   реально применяет execution layer, не обычная функция-подсказка.
4. Передать исходный monotonic deadline **после вычитания** резерва outer,
   production-refit и publication; не создавать новый бюджет на каждом fold.
   Все четыре fit и guard входят в original task deadline. Не запускать OOF,
   если резерв не остаётся. Не обрезать стандартный solver budget ради пилота.
5. Заморозить весь JSON-wrapper/его hashes в recipe до outer exposure. После
   допуска не делать retune на A+B. Outer и production labels не выбирают
   множители. Core самостоятельно делает allowed→outer и полный train→test refit.
6. После refit восстановить точный probability-column order и применить
   `predict_with_rule(p_test, returned_classes, wrapper['rule'])`. Проверить
   сохранённый rule hash; model.classes_ может отличаться порядком. Identity
   остаётся schema-valid fallback; структурные ошибки не скрывать как успех.

```python
from agent.oof_classification import fit_oof_guarded_rule
from agent.classification_decision import predict_with_rule

wrapper = fit_oof_guarded_rule(
    X_outer_fit, y_outer_fit, classes,
    groups=raw_groups, split_policy='grouped',
    tuning_rows=A, selection_rows=B, folds=three_A_only_folds,
    fit_predict=bounded_probability_callback,
    fold_timeout_seconds=fold_cap,
    deadline=original_deadline_minus_final_reserve,
    clock=monotonic_clock,
)
# Freeze JSON and recipe here. Refit model in host, then apply without tuning:
labels = predict_with_rule(p_after_refit, model_probability_classes, wrapper['rule'])
```

Эти hooks описаны для владельца core; соответствующие файлы не патчились.

## Команды повторения и артефакты

Из checkout с замороженными runtime source hashes и установленным requirements.
Каждое новое имя каталога используется один раз; повторный run в том же каталоге
отклоняется. На Windows пути можно записать нативно. Проверка -O ниже не обучает.

```bash
python experiment_oof_classification.py --prepare-only --output-dir data/experiments/oof_classification/repro-v1-NEW
python experiment_oof_classification.py --run-prepared data/experiments/oof_classification/repro-v1-NEW
python experiment_oof_classification.py --re-audit data/experiments/oof_classification/repro-v1-NEW
python -O experiment_oof_classification.py --re-audit data/experiments/oof_classification/frozen-v1-20261021
```

При доступном подлинном public task добавить `--public-task /absolute/task.md`
только на prepare. Он будет известной sanity-проверкой, не confirmation;
production-refit на полном public TRAIN предсказывает TEST, но без его ответов
production metric не объявляется. Сырые артефакты этой серии находятся в
`data/experiments/oof_classification/frozen-v1-20261021/`: plan, source.joblib/CSV,
splits, report, frozen wrappers, requests, receipts, models, .npy probabilities,
CSV predictions и независимые audits. В Git включён только лёгкий отчёт.

```text
preregistration.json: ca8c8c1a94008fd0ab0cc4f0b6285cab369745cce99aff89a05c16b50f00b627
report.json: a27be626e11bb30b16c1b704fc9101ae37cd79709540beeee43cead11bbd388f
reaudit.json: ced56b2ee9accecad5c4655cfed0179a6a23965e7693d0bd1d7a7df42f247668
independent_results_audit.json: bacf8206e51eee8d202fcd890557934e511dcd750f95e5f3312d19c5c39909a4
independent_split_audit.json: 117a5fc65e0887ff62368353696326e9aeda7c366c07b1fc926bb02b1e7cdc79
source_sha256:
agent/oof_classification.py: 99bdbf3ef036259385fe92602fa614867c9dbe8137fdd7e8ea8ec4ba683529f3
agent/classification_decision.py: e81dc5ca13483ea55dc5f5d3798868fe643e4d64d01bbe82400ee54698cded30
experiment_oof_classification.py: 670343074aaca46915aac100a75c12ea6280137e9b5b0c73adb10225c5e1d7c2
oof_classification_cases.py: 6bc37db58075749566f708facf0e199e7cdd0a379696607f518cd4f0ea44b0ef
agent/execution.py: b2d576f765941b392bd470d98ce0a420e9ae00c533723e225e48005f003085b5
agent/resources.py: e5cdd4fc35276caf3a82672306441bbc47364eabad719b11dd5234aa1a065306
agent/file_io.py: fc374b6a628e97f47499b5baa720344a783e905707b5cccc297d4c4aeb1a33c9
```

Конкурсный API: 0 запросов; `.env` не читался; platform submissions: 0;
локальный публикационный ZIP не создавался и не заменялся; merge/force push: 0.
Стандартный CI выполнял свою штатную проверку сборки архива в одноразовом
checkout, без публикации или отправки этого архива на платформу.
