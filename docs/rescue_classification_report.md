# Classification rescue v2: измерения и ограничения

**Решение: сохранить incumbent; кандидата для включения по умолчанию нет.** На трёх заранее заданных grouped dev-срезах средний Macro F1 фиксированного incumbent — **0.6179357399**, CatBoost `SqrtBalanced` — **0.6168863151** (paired Δ **−0.0010494249**), LightGBM без весов — **0.6175072865** (Δ **−0.0004284534**). Оба финалиста не достигли полезного порога +0.005, имеют падение >0.01 на отдельном срезе и существенные классовые провалы. Лучший средний результат среди двух новых финалистов у LightGBM; его full-TRAIN refit и локальный TEST CSV подтверждены только как проверка доставки. Общие scheduler/router-файлы не изменены, подключение имеет статус `integration_pending`; на платформу ничего не отправлено.

## Что и на каком коде проверено

Incumbent — свежий `origin/main`, SHA **`d05572846369f3b1e9cfa33f4008dda028db4fe0`**; он совпал с исходным SHA задания. Работа ведётся в отдельном checkout, ветка `work/rescue-classification-v2`. Для native incumbent создан отдельный неизменный detached worktree этого SHA. Исходный main запускался обычным `solution.py --mode offline --time-budget 900 --research-search adaptive` без monkeypatch общих модулей.

План зафиксирован до обучения: `docs/rescue_classification_plan.md`. Пять рецептов проходят screening только на seed 42 (максимум 10 модельных обучений: internal fit + fresh refit каждого). Два лучших проходят seed 43 и 44. Полезный порог — средний paired-прирост +0.005; падение Macro F1 >0.01 на срезе или F1 класса >0.05 требует расследования и исключает default-рекомендацию. Это инженерные пороги, не статистическая гарантия. Все срезы — development evidence, включая исторически изученный seed 42.

Изолированные кандидаты используют существующие `make_boosting_recipe` / `make_xgboost_recipe`, source-фабрики и настоящий `freeze_candidate` → `run_candidate`; модельный код не продублирован. Timeout каждого — 300 с, CPU, 8 потоков, максимум 2 000 итераций, patience 50. Предобработка и число деревьев выбираются только внутри текущего fit, затем идёт новый refit на всех его строках. Внешние ответы остаются в host и не входят в `X_predict`.

## Данные, дубли и разбиения

Официальные файлы получены штатным `prepare_data.py classification`, его проверка прошла. TRAIN: **87 670 строк, 42 признака**, из них 39 числовых и 3 категориальных. `proto/service/state` имеют 133/13/9 значений. Чтение контрактом сохраняет исходные строки (`object`), преобразование типов происходит внутри fit. Пустых/NULL признаков и нечисловых/бесконечных числовых значений не обнаружено; констант нет. `is_ftp_login` и `ct_ftp_cmd` — точные дубли столбцов; в этом screening они сохранены одинаково для всех кандидатов.

Есть **54 228 групп одинаковых raw-признаков**, 5 816 групп дублей с 39 258 строками; сверх первых экземпляров — 33 442 строки. В **1 458 группах** одинаковым признакам соответствуют разные метки; в них **14 829 строк**. Например, одна группа из 90 строк содержит DoS 34, Exploits 33, Reconnaissance 6, Backdoor 6, Analysis 5, Fuzzers 4, Generic 2. Это ограничивает различимость таких строк, но не задаёт вычисленный потолок Macro F1.

| Класс | Весь TRAIN | Строки в противоречивых группах | Dev support 42 / 43 / 44 |
| --- | ---: | ---: | ---: |
| Analysis | 1 000 | 709 | 192 / 221 / 194 |
| Backdoor | 873 | 634 | 192 / 166 / 182 |
| DoS | 6 132 | 4 838 | 1 212 / 1 236 / 1 215 |
| Exploits | 16 696 | 6 503 | 3 296 / 3 385 / 3 323 |
| Fuzzers | 9 092 | 939 | 1 803 / 1 795 / 1 782 |
| Generic | 20 000 | 166 | 4 150 / 3 915 / 3 677 |
| Normal | 28 000 | 161 | 5 665 / 5 571 / 5 567 |
| Reconnaissance | 5 245 | 877 | 881 / 985 / 1 114 |
| Shellcode | 567 | 1 | 108 / 101 / 119 |
| Worms | 65 | 1 | 16 / 12 / 16 |

Разбиения построены штатной grouped-политикой по признакам **без ID/target**, до обучения. Менялся только внешний seed; внутренний/model seed всегда 42.

| Dev seed | Fit | Held | Fit groups | Held groups | Пересечение групп |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 70 155 | 17 515 | 43 384 | 10 844 | 0 |
| 43 | 70 283 | 17 387 | 43 388 | 10 840 | 0 |
| 44 | 70 481 | 17 189 | 43 386 | 10 842 | 0 |

Внутри seed-42 fit все пять кандидатов использовали одинаковое дополнительное разбиение 55 945/14 210 с нулевым пересечением групп. Официальный TEST (41 166 строк) не используется для выбора рецепта; меток TEST нет.

## Native incumbent: достигнутый маршрут

Неизменный main завершился за **816.747554 с** при лимите solver 900 с. Получен новый CSV, outer/refit receipts и хеш публикации проверены; API attempts/responses — **0/0** из настоящего `run.json`.

Опубликован baseline — фиксированное равное усреднение вероятностей двух моделей: LightGBM (`n_estimators=400`, `num_leaves=15`, `reg_lambda=3`, `class_weight="balanced"`, остальные штатные настройки) и RF (`n_estimators=120`, `max_depth=18`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced_subsample"`). Точный JSON сохранён вместе с receipt; его SHA-256 — `2f713e0d8c4d76b53f6fdb180a08ae99e20489ce84f7a074532c28a366511e02`.

Внутренний outer acceptance предпочёл baseline: 0.6207966606 против 0.6085935474 у выбранного в INNER кандидата. Это **другая, вложенная проверка**, не внешняя оценка 0.6270856863. После этого baseline переобучен на всех 70 155 строках derived TRAIN и предсказал 17 515 внешних dev-строк.

Warmup выполнил 16 проверенных trials; TPE/adaptive proposals — **0**. Hook extensions был вызван: frequency попытался исполниться и завершился ошибкой `candidate_execution_failed` (`ValueError`); boost-CB, boost-LGB, ensemble и AutoGluon пропущены из-за `finalization_reserve`. Поэтому это не эмпирическое сравнение «одна длинная модель против множества TPE trials»: до TPE маршрут не дошёл. XGBoost в штатный router main не подключён. Изолированный опыт ниже отдельно подтверждает его действительное обучение.

## Screening: все пять исходов

Порядок запусков соответствует строкам таблицы. Время — весь изолированный вызов с подготовкой/проверкой host, а не только `.fit()`. Все пять получили `completed`, host-проверенные предсказания и `fresh_full_refit=true` на 70 155 строках; process timeout/error/skip не было. Это full-fit внутри dev-опыта, ещё не refit на всём официальном TRAIN.

| Рецепт | Macro F1, seed 42 | Δ к native incumbent | Время, с | Internal / selected iterations |
| --- | ---: | ---: | ---: | ---: |
| CatBoost depth 6, без весов | 0.5115631043 | −0.1155225820 | 186.735511 | 341 / 339 |
| CatBoost depth 6, `SqrtBalanced` | **0.6309870625** | **+0.0039013763** | 174.982755 | 361 / 319 |
| LightGBM 31 лист, без весов | 0.6158309260 | −0.0112547602 | 53.092247 | 367 / 318 |
| XGBoost `depth6_v1` | 0.6015113897 | −0.0255742965 | 37.756223 | 458 / 408 |
| XGBoost `depth4_regularized_v1` | 0.5896040243 | −0.0374816619 | 58.445714 | 1 028 / 978 |

У CatBoost внутреннее обучение заняло 79.805140 и 80.196617 с; разрешённая доля invocation — 0.27521895. До лимита 2 000 итераций обе модели не дошли, после лучшей итерации прошли только 2 и 42 шага при patience 50. С учётом реализации callback это указывает на остановку внутренним временным бюджетом, а не на естественную сходимость. Следовательно, результат относится к **ограниченному 300-секундному рецепту**; не доказано, как поведёт себя более длинный CatBoost. Различие весов также сопровождается разным числом выбранных деревьев, поэтому это сравнение полных рецептов, не изолированный причинный эффект весов.

XGBoost receipts прямо указывают `stopped_by="early_stopping"`, метрику `mlogloss` и `objective="multi:softprob"`: depth 6 выбрал 408 из 458 обученных итераций; depth 4 — 978 из 1 028. Оба полностью refit и закончили задолго до deadline. Их проигрыш на этом срезе подтверждён **достигнутыми моделями**, а не отсутствием hook; продолжение до 2 000 деревьев произвольно после просмотра dev-ответов не выполнялось. Macro F1 независимо пересчитан host по сохранённым меткам, LogLoss не выдаётся за целевую метрику.

По заранее заданному правилу дальше проходят только **CatBoost SqrtBalanced и LightGBM**. Остальные после первого среза остановлены.

## Какие ошибки меняются

| Класс | Support | Main precision | Main recall | Main F1 | CB Sqrt F1 | LGB F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Analysis | 192 | 0.104496 | 0.447917 | 0.169458 | 0.287356 | 0.220339 |
| Backdoor | 192 | 0.083426 | 0.390625 | 0.137489 | 0.242424 | 0.265060 |
| DoS | 1 212 | 0.375360 | 0.429868 | 0.400769 | 0.473616 | 0.241152 |
| Exploits | 3 296 | 0.874198 | 0.537621 | 0.665790 | 0.661657 | 0.746002 |
| Fuzzers | 1 803 | 0.635036 | 0.868552 | 0.733661 | 0.700317 | 0.737314 |
| Generic | 4 150 | 0.998281 | 0.979277 | 0.988688 | 0.988545 | 0.989307 |
| Normal | 5 665 | 0.989326 | 0.834422 | 0.905295 | 0.890358 | 0.930563 |
| Reconnaissance | 881 | 0.812428 | 0.801362 | 0.806857 | 0.792170 | 0.807858 |
| Shellcode | 108 | 0.522472 | 0.861111 | 0.650350 | 0.536585 | 0.660714 |
| Worms | 16 | 0.812500 | 0.812500 | 0.812500 | 0.736842 | 0.560000 |

Main предсказывает Analysis 823 раза и Backdoor 899 раз при 192 реальных строках каждого: много ложных срабатываний. CB Sqrt повышает precision этих классов до 0.320513/0.717949, но recall снижается до 0.260417/0.145833. Для DoS recall растёт до 0.751650, precision падает до 0.345731. Улучшение общего F1 сопровождается **−0.113764 F1 Shellcode и −0.075658 F1 Worms** против main, превышая классовый порог. У Worms CB нашёл 14/16 вместо 13/16, но дал больше ложных срабатываний: одна лишь recall скрыла бы ухудшение F1.

Основные пары ошибок main: Normal→Fuzzers 814 (14.37% Normal), Exploits→DoS 557 (16.90%), Exploits→Backdoor 398 (12.08%), Exploits→Analysis 322 (9.77%), DoS→Backdoor 290 (23.93%), DoS→Analysis 222 (18.32%). Полная confusion matrix 10×10, precision/recall/F1/support всех кандидатов и диагностические примеры конфликтов сохранены в JSON.

Подтверждены проблемы редких классов и конкурирующих решений между DoS/Exploits/Analysis/Backdoor. Без весов CatBoost распознаёт лишь 39/1 212 DoS и 0/16 Worms. Проверенные одиночные рецепты не дали устойчивого улучшения incumbent. Новые признаки, частотные преобразования, OOF-настройка решений и ансамбли в этом ограниченном screening не проверялись; это не отрицательные результаты для этих методов.

## Повторные срезы и решение

| Seed | Фиксированный incumbent | CB Sqrt | Δ CB | LGB | Δ LGB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6270856863 | 0.6309870625 | +0.0039013763 | 0.6158309260 | −0.0112547602 |
| 43 | 0.6077597140 | 0.5967365845 | **−0.0110231296** | 0.6141759607 | +0.0064162467 |
| 44 | 0.6189618195 | 0.6229352982 | +0.0039734787 | 0.6225149727 | +0.0035531532 |
| Среднее | **0.6179357399** | 0.6168863151 | **−0.0010494249** | 0.6175072865 | **−0.0004284534** |

Контрольные replay seed 43/44 завершились за 22.243128/22.241539 с. CB Sqrt seed 43 — за 79.155134 с, 195 внутренних / 145 выбранных итераций; от лучшей прошло полных 50 шагов patience. Поэтому его проигрыш нельзя объяснить тем же внутренним временным ограничением, которое действовало на seed 42. LGB seed 43 — за 29.857413 с, 144/95 итераций. На seed 44 CB Sqrt занял 167.706475 с и выбрал 291 из 341 итерации; LGB — 30.411142 с и 120 из 169 итераций. Во всех четырёх повторах подтверждён новый refit на полном fit соответствующего среза (70 283 или 70 481 строк); process timeout отсутствует.

На seed 43 у CB Sqrt F1 Shellcode 0.508143 против 0.648464 контроля (−0.140321), Worms 0.640000 против 0.709677 (−0.069677). LGB имеет положительную общую разницу на этом срезе, но F1 DoS падает с 0.381061 до 0.216405 (−0.164656). На seed 44 у CB Sqrt F1 Shellcode 0.584958 против 0.699670 (−0.114712; support 119); у LGB F1 DoS 0.177552 против 0.402032 (−0.224480; support 1 215). LGB относит 1 030 из 1 215 DoS к Exploits. Удачный общий F1 одного среза не снимает эти классовые провалы.

CB Sqrt нарушает порог падения на seed 43 (−0.0110231296), LGB — на seed 42 (−0.0112547602). Средние разницы обоих отрицательны, а классовые ухудшения повторяются. Поэтому incumbent сохраняется; победа LGB среди двух новых финалистов не является основанием заменить incumbent. Повышение native-бюджета ради этих рецептов не рекомендовано после провала заранее заданных условий.

Контроль для повторов — **точный опубликованный native recipe**, замороженный после первого запуска. На seed 42 сохраняются его исходные native-предсказания; на 43/44 он обучается заново через изолированный worker с cap 300 с, без нового выбора рецепта. Это сравнение с фиксированным incumbent-рецептом, не три повторных запуска полного main scheduler; исходные 900 с выбора рецепта и времена replay сохраняются раздельно.

## Full-TRAIN refit: проверка доставки

Для `lightgbm_none` выполнен отдельный fresh refit на **87 670 строках TRAIN**, получены **41 166 предсказаний TEST** за **42.887199 с**. Внутренний grouped split 70 155/17 515 имеет overlap 0; из 200 внутренних итераций выбрана 151 по `host_Macro_F1`. При финальном refit весь TRAIN разрешён для внутреннего выбора числа деревьев; TEST-метки отсутствуют. Source и рецепт совпадают с измеренными dev-кандидатами, модель заново обучена на всех строках TRAIN.

Host подтвердил формат, допустимые классы и исходный порядок TEST ID; receipt содержит `fresh_full_refit=true`, `full_fit_rows=87670`, `predict_rows=41166`. Статус публикации — `local_verified_csv_written`; файл `data/experiments/rescue_classification_v2/trials/lightgbm_none_full_refit/final_predictions.csv`, SHA-256 **`7cbbb8752f5c7e7b92f56e7fc9bfc4559a5c04f7bc21ce52ec8c29684c6349f8`** совпадает с worker output. Это **unscored TEST**, локальная проверка доставки, без отправки на платформу и без изменения решения о качестве.

Same-scheduler off/on опыт не выполнялся: общего integration SHA нет. Возможная будущая опция LGB остаётся выключенной; точный контракт и ограниченная предлагаемая проводка описаны в `docs/rescue_classification_integration.md`.

## Ресурсы и доказательства

Python **3.12.14**; NumPy 1.26.4, SciPy 1.12.0, pandas 2.2.3, sklearn 1.5.2, CatBoost 1.2.8, LightGBM 4.6.0, XGBoost 2.1.4, Optuna 4.2.1, psutil 6.1.1 — соответствуют официальным pins. Полный перечень установленных/отсутствующих пакетов записан в плане; AutoGluon отсутствует и не исследовался. Зависимости в процессе решения не устанавливаются, официальный список не изменён.

Host cgroup: 8 CPU, **8 GiB RAM**, а не конкурсные 32 GB. Supervisor запросил tree-RSS cap 6 GiB, однако штатный монитор сообщил `RuntimeError: process inspection unavailable`: namespace процессов не совпадает с `/proc`. Поэтому во всех этих опытах **peak RSS = null, 6 GiB tree-cap не подтверждён и не применён, cleanup не подтверждён**. Фактическое ограничение — общий hard cgroup 8 GiB; конечные wall-clock deadlines действовали. Значение `memory_limit_exceeded=false` при недоступном мониторе не доказывает расход ниже 6 GiB. Квалификация конкурсного ресурса не заявляется.

Артефакты не коммитятся вместе с данными:

- `data/experiments/rescue_classification_v2/plan.json`: версии, source/input hashes, полные recipes, заранее фиксированные split-индексы, class support, диагностика. SHA-256 плана: `2d308b5eee8728d618d4e63791ad4f8ed7c2f8da509b42ce7e722a54559d814e`.
- `data/experiments/rescue_classification_v2/experiments.json` и `trials/*/record.json`: машиночитаемая таблица всех исходов; рядом реальные CSV, host receipts, model evidence и stdout/stderr.
- `data/experiments/rescue_classification_v2/selection_summary.json`: итог `retain_incumbent`, все paired-разницы и нарушения трёх условий качества.
- `data/experiments/rescue_classification/native_incumbent/report.json`, `execution.json`, `task/.agent_runs/*/run.json`: native маршрут, измерения, outer acceptance и publication. SHA-256 native плана: `0726323a92050a74dcbd8ed7bd8aa5c911782adba43ced339cefbafde01e36aa`.
- SHA-256 официального TRAIN: `171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`; TEST: `b825449603931b5f05a8855e3d92b57775221c93ed4730ac6b4c75b9ea608862`. Хеши остальных входов находятся в плане, исходные CSV не изменялись.

Native API 0/0 подтверждены счётчиками host. У изолированных trials нули имеют явное происхождение `structural_no_provider_path`: стенд/кандидаты не создают API-клиент; эти поля не притворяются измеренными native-счётчиками. Финальный повторный `audit` проверил **13/13 completed записей**: 12 dev-результатов и один unscored full-TRAIN refit. Проверены сохранённые хеши, receipts, формат публикации и пересчитаны dev-метрики.

После обучения исправлена только воспроизводимость стенда: `compare-incumbent` теперь различает SHA checkout эксперимента и SHA фиксированного main, дополнительно проверяя все хеши runtime-файлов. Коммит документации/стенда больше не выдаётся за смену incumbent. Snapshot исходного стенда — `data/experiments/rescue_classification_v2/harness_at_run.py`, SHA-256 `165f7eca649cd6255d6f4ea2bfaa839b2da37d0d85372b9b203949444afce0cc`; итогового `experiment_rescue_classification.py` — `863b0ef94cc25df6dd0326f93eceaf26a0c65af6507bf8a4c0217656895709bc`. Запечатанные планы, source кандидатов, рецепты и измеренные результаты не переписаны; повторный audit после исправления также прошёл 13/13.

Проверки: **199 тестов за 9.627 с** (три новых модуля стенда и существующие boosting candidate/native, XGBoost candidate, classification decision, OOF classification, frequency candidate). После исправления воспроизводимости добавлены два regression-теста; повторный прогон трёх новых модулей — **25 тестов за 0.062 с**, успешно. Всего проверен **201 различный тест**.

Для native-запуска также сохранён исходный `native_incumbent/harness_at_run.py`. После запуска усилены проверки идентичности outer/refit receipts и исходных входов в audit-части native-стенда; обучение не повторялось. Его `report.json` различает хеши стенда запуска и проверки. Оба снимка включены в архив.

## Точное воспроизведение

Оба `--help` проверены. Запускать последовательно из checkout этой ветки, с уже подготовленным Python 3.12 окружением официальных версий. Имена evidence-папок должны быть новыми; после `prepare` не менять запечатанные source-файлы/окружение. Создание worktree выполняется один раз; при существующем пути использовать другой новый путь и передать его в `--source-root`.

Linux:

```bash
git worktree add --detach ../classification-incumbent d05572846369f3b1e9cfa33f4008dda028db4fe0
.venv/bin/python prepare_data.py classification
.venv/bin/python experiment_rescue_classification_native.py --prepare --output data/experiments/repro_native --source-root ../classification-incumbent
.venv/bin/python experiment_rescue_classification_native.py --run-prepared --output data/experiments/repro_native
.venv/bin/python experiment_rescue_classification_native.py --audit --output data/experiments/repro_native
.venv/bin/python experiment_rescue_classification.py prepare --mode offline --output data/experiments/repro_screen --timeout 300 --memory-gib 6
.venv/bin/python experiment_rescue_classification.py screen --mode offline --output data/experiments/repro_screen
.venv/bin/python experiment_rescue_classification.py compare-incumbent --mode offline --output data/experiments/repro_screen --incumbent-output data/experiments/repro_native
.venv/bin/python experiment_rescue_classification.py repeat --mode offline --output data/experiments/repro_screen
.venv/bin/python experiment_rescue_classification.py full-refit --mode offline --output data/experiments/repro_screen --candidate lightgbm_none
.venv/bin/python experiment_rescue_classification.py audit --mode offline --output data/experiments/repro_screen
```

Windows PowerShell (те же реализованные параметры; запуск Windows в этой работе не измерялся):

```powershell
$classificationPython = '.\.venv\Scripts\python.exe'
git worktree add --detach ../classification-incumbent d05572846369f3b1e9cfa33f4008dda028db4fe0
& $classificationPython prepare_data.py classification
& $classificationPython experiment_rescue_classification_native.py --prepare --output data/experiments/repro_native --source-root ../classification-incumbent
& $classificationPython experiment_rescue_classification_native.py --run-prepared --output data/experiments/repro_native
& $classificationPython experiment_rescue_classification_native.py --audit --output data/experiments/repro_native
& $classificationPython experiment_rescue_classification.py prepare --mode offline --output data/experiments/repro_screen --timeout 300 --memory-gib 6
& $classificationPython experiment_rescue_classification.py screen --mode offline --output data/experiments/repro_screen
& $classificationPython experiment_rescue_classification.py compare-incumbent --mode offline --output data/experiments/repro_screen --incumbent-output data/experiments/repro_native
& $classificationPython experiment_rescue_classification.py repeat --mode offline --output data/experiments/repro_screen
& $classificationPython experiment_rescue_classification.py full-refit --mode offline --output data/experiments/repro_screen --candidate lightgbm_none
& $classificationPython experiment_rescue_classification.py audit --mode offline --output data/experiments/repro_screen
```

Команды full-refit выше воспроизводят проверку доставки экспериментального LGB; они не включают его в main.

Переносимый архив `rescue-classification-evidence.zip` содержит scripts, `evidence_manifest.json`, dev-ответы, изолированные результаты и traces; исходные датасеты исключены. Для независимой проверки из корня распакованного архива в Linux или PowerShell достаточно Python со стандартной библиотекой, без ML-пакетов:

```text
python scripts/experiment_rescue_classification_rescore.py .
```

Команда пересчитывает 12 dev-оценок и проверяет один unscored TEST CSV; она не запускает обучение и не отправляет предсказания.
