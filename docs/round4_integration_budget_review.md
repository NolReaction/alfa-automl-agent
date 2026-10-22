# Раунд 4: разбор времени и допуска кандидатов

Дата: 23 октября 2026, МСК. База нового контроля: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`.

Анализ только чтением, без ML/API и без изменения репозитория. Все пять сохранённых `native/run.json` побайтово совпадают с исходными `/tmp`-отчётами из wrapper-путей. Существующий `experiment_rescue_integration_report.py` использован для `native_report_index.json`; это индекс, а не независимый аудит. Дополнительные расчёты находятся в `five_run_analysis.json`, воспроизводящий скрипт — `analyze_saved_round4.py`.

## Время без двойного счёта

В таблице три непересекающихся интервала из `time_budget_seconds`, `local_research.started_remaining_seconds`, `ended_remaining_seconds` и итогового `elapsed_seconds`. Первая колонка времени включает startup и измерение baseline INNER. Вторая включает hook расширений. Последняя включает freeze, OUTER, refit, публикацию. Их сумма равна native elapsed. Длительность baseline subprocess и отдельных fits — вложенные интервалы, к этим суммам их не прибавляем.

| Прогон | До local, с | Local включительно, с | После local, с | Native всего, с | Hook внутри local, с | Остаток native дедлайна, с |
|---|---:|---:|---:|---:|---:|---:|
| classification_main_900 | 226.289 | 445.704 | 124.069 | 796.062 | неизвестно; hook не вызван | 97.718 |
| classification_priority_900 | 224.857 | 586.189 | 50.882 | 861.928 | 361.645 | 31.826 |
| regression_main_900 | 105.781 | 362.788 | 61.237 | 529.805 | неизвестно; hook вызван без timestamps | 363.669 |
| regression_integrated_off_900 | 90.647 | 379.677 | 70.598 | 540.922 | 50.311 | 352.749 |
| regression_integrated_on_900 | 91.727 | 342.066 | 74.970 | 508.763 | 57.584 | 384.729 |

Нативные `stages` содержат id/status, но не измеренные durations. `time_allocation.stage_seconds` — эвристические оценки, не фактическая длительность стадий. `elapsed_seconds`, `host_elapsed_seconds`, feedback.elapsed и measurements дублируют/включают работу друг друга; суммировать их все нельзя. Wrapper elapsed отдельно включает работу за границами native таймера.

## Расширения: вызов, проверка и публикация различаются

- **Classification main:** 13 ordinary warmup attempts, 12 verified, 1 improvement; adaptive 0; остановка `finalization_reserve`. Hook не вызван; research_extensions отсутствует. Это отсутствовавший запуск расширений, а не их нулевое качество.
- **Classification priority:** hook `before_warmup`; verified LGBM 46.594 с, CatBoost 264.435 с, frequency 49.926 с. Их feedback durations суммарно 360.955 с внутри hook 361.645 с; разница включает подготовку/записи. LGBM INNER .607614 и CatBoost .509536 не лучше baseline .615424. Frequency INNER .620115 лучше, но `refit_budget`: estimate 316.178 > remaining 307.613. History пропущена `regression_required`; ensemble — `ordinary_measured_finalist_required`; AutoGluon — `autogluon_unavailable`. Затем ordinary warmup 9/8 verified, adaptive 0.
- **Regression main:** обычный warmup 14 + adaptive 50 = 64, причина остановки `trial_limit`. Extensions запускают frequency, history, CatBoost, LGBM, ensemble (все verified); history принят. AutoGluon пропущен как unavailable. Timestamps hook отсутствуют, поэтому нельзя извлечь полную длительность hook из единого сохранённого интервала.
- **Regression off:** hook до warmup: history, LGBM, CatBoost, frequency (4 verified, 1 accepted). Ensemble пропущен `ordinary_measured_finalist_required`, AutoGluon unavailable. Далее ordinary 64/64, без нового improvement, стоп `trial_limit`.
- **Regression on:** тот же порядок слотов, но rescue_regression заменяет CatBoost; history и rescue приняты (4 verified/2 accepted). Rescue INNER 32.163920 за 5.065 с, history INNER 33.794489. Ensemble пропущен из-за отсутствия ordinary measured finalist в момент раннего hook, а не из-за нехватки времени. Далее ordinary 64/64, без улучшений, стоп `trial_limit`.

## INNER winner и опубликованное

| Прогон | Лучший измеренный INNER | Допущенный selection | Frozen/published | Native OUTER candidate/baseline |
|---|---:|---:|---|---|
| Classification main | .6185327725 | .6185327725 | ordinary LGBM, candidate | .6283537030 / .6207966606 |
| Classification priority | frequency .6201153329 | baseline .6154239385 | frozen_candidate отсутствует; baseline | null / .6207966606 |
| Regression main | history 33.79448887 | 33.79448887 | history, candidate | 44.56965721 / 49.02132404 |
| Regression off | history 33.79448887 | 33.79448887 | history, candidate | 44.56965721 / 49.02132404 |
| Regression on | rescue 32.16391985 | 32.16391985 | rescue, candidate | 39.47145826 / 49.02132404 |

В classification priority также измерен **тот же ordinary LGBM**, который main опубликовал: INNER .6185327725, но отклонён `refit_budget` (estimate 221.893 > remaining 174.381). Это два конкретных кандидата с лучшим INNER, не допущенные по времени. Нельзя описывать этот arm просто как «новые модели хуже baseline».

Native OUTER — внутренний финальный выбор модели, он помечен `selection_uses_outer=true`, `unbiased_estimate=false`. Это не отдельный sealed public TRAIN holdout. Сохранённые внешние classification CSV дают .6221456784 и .6270856863; эти цифры не совпадают с native OUTER по назначению. Старый вывод о ранней политике нельзя вывести из одной native метрики или суммы длительностей.

## Повтор frozen source + recipe

В исходных `/tmp` изучены все native-v3 `context.json` (18/15/74/73/73 контекста в перечисленных прогонах). **Повторений `(code_hash, recipe_hash, phase)` внутри одного прогона нет.** Повтор одной пары hashes есть только у baseline/победителя на `inner_selection → outer_validation → full_refit`. Это разные fit/predict наборы и обязательная цепочка, а не доказанная повторная тренировка на том же split. Один отсутствующий receipt в каждом classification arm соответствует failed CatBoost warmup (`CandidateExecutionError`); из одного отсутствия receipt точный вид сбоя не устанавливается.

`local_research.duplicates` = 10/9/12/12/12 — счётчик рецептов, пропущенных до вызова evaluator, а не число впустую выполненных fits. Нет основания объявлять кэш одинаковых frozen INNER-вызовов уже доказанной экономией. Этот анализ не исключает семантически одинаковые модели с различными hashes и не измеряет возможную экономию кэширования компонентов/OOF.

## Резерв финализации и первый 3600-секундный замер

- У классификации давление бюджета доказано stop reason, отсутствием adaptive/hook и двумя конкретными `refit_budget` rejection. У priority финализация начинается с remaining 82.610 с, заканчивается через 50.783 с; main — с 221.686 с, заканчивается через 123.968 с.
- У регрессии остановка доказанно **trial_limit**, не резерв. В трёх прогонах остается 352–385 с native бюджета. Простое увеличение CLI-бюджета до 3600 не гарантирует больше ordinary trials при cap64; запуск на более длинном бюджете может изменить startup/рецепты, поэтому его elapsed/качество не экстраполируем из 900.
- Поле minimum_final_reserve ≈268 с — исходный floor; diagnostics говорят `reserve_policy=measured_pending_recipes` с применённым floor ≈17.87 с. В конце report final_reserve — повторная эвристическая оценка всей цепочки по уже накопленным measurement, не запись точного резерва в момент раннего отказа. Из отношения «итоговая оценка / фактическая финализация» нельзя само по себе доказать ошибку аллокатора.

**Первым native-замером на 3600 с выбрана классификация на базе e62fea1 с сохранённым поздним hook**, чтобы измерить, исчезает ли starvation: достигнутые adaptive trials/extensions, лучшие INNER, причины admission/rejection, frozen hashes и фактический опубликованный CSV. Не включать общий ранний hook только на основании опыта900: он съел 264 с CatBoost и потерял допуск обоих лучших кандидатов. Если первым отдельным кодовым экспериментом выбирается использование свободного времени, регрессия даёт прямой тест budget-aware ordinary cap64; это отдельная гипотеза, её не следует смешивать с классификационной перестановкой расширений.

RSS/полнота cleanup для этих прогонов остаются unknown (несовпадающие PID namespaces); успешный native статус и настоящий elapsed их не квалифицируют. Сохранённые старые audits используются как документы; настоящая записка не выполняет заново полный receipt/resource/CSV-метрический аудит.


## Исправление аллокатора отдельно от наблюдений

Ошибка воспроизведена напрямую на `BudgetPolicy.finalization_timeouts`: старый алгоритм сначала выдавал baseline полные floors, а затем делил с baseline и остаток, необходимый кандидату. При достаточном суммарном времени candidate caps оказывались меньше собственных estimates. Это самостоятельный аргумент к исправлению; таблица старых native timing не доказывает, что именно эта ошибка испортила их результат.

Коммит `8a88dc348f13510a92dfbcfbfc7cafb88417974c` сначала покрывает все capped estimates, затем делит surplus. При настоящем дефиците прежний приоритет baseline сохранён. Ошибка вычитания абсолютных monotonic timestamps на границе резерва обрабатывается допуском до четырёх ULP исходного deadline, максимум 1 микросекунда; поправка вычитается из крупнейшей доли, не добавляется к бюджету.

68 focused tests (budget, coordinator, CLI rollout, entry recovery) прошли за 1.726 секунды. Native пара и полный аудит в момент записи ещё выполняются; её результаты относятся к отдельному разделу текущего статуса после завершения. Runtime единственного candidate arm отличается от immutable контроля только `agent/research_budget.py`.
