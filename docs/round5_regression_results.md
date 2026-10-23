# Round5 regression: сохранить действующий rescue

Решение: **retain-incumbent (`rescue_regression_v1`)**. Новый family `round5_regression_ensemble_v1` остаётся изолированным исследовательским кодом. Новый алгоритм не рекомендуется к активации; freeze победителя и full-TRAIN refit не выполнялись. Общий runtime, routing, scheduler, действующий rescue и существующие helpers не изменены.

E75 выиграл RMSE на двух из трёх confirmation-переходов и снизил pooled RMSE на **0.7233%**, суммарную SSE на **1.4414%**. Однако Residential на period2 ухудшился по SSE на **18.6023%**: 6449 строк, прирост **1,883,216.56**. Это нарушает заранее зафиксированный сегментный порог. LGB нарушил предел fold RMSE на period2 и получил существенные сегментные потери на period2 и period3; дальнейший confirmation period4 остановлен. Numeric-only варианты не прошли screen.

## Идентичность и протокол

- База: `66e9481cb61cb83cfecab53be6984d7d2bf829b3`. Main при заключительной проверке оставался на этой базе; rebase не выполнялся.
- План до первого нового fit: локальный `8f85aa5540844ef67b9e42db4f988e0c18bb4650`, опубликованный эквивалент `38fca61c1ad0581e3e6eb030b363d8bff56c3db5`.
- **Фактически измеренный implementation SHA:** `c0ba45118c41dfa4b0ad0360f925874692dd3082` (clean).
- **Опубликованный эквивалент implementation:** `41f0b09c766682515684d4838c6c91dfa95bb154`.
- Их идентичное Git tree: `aab125446d87e843aef9e0ca4ad83ba4518dd31f`. SHA commits различаются из-за метаданных публикации через GitHub connector; байты дерева сверены. Последующий commit отчётов меняет только документацию.
- Совокупный SHA256 измеренных зависимостей: `1e5f32d785291944115b9216ae4a0bd3020424835a506d0e8a940c89cb858478`. Полные file-level hashes в JSON и archive manifest.

Все восемь screen arms были заданы заранее: I, LGB, E25, E50, E75, N, N25, N50. Screen: period4, 180 секунд на целую invocation. Confirmation: свежие I и максимум два финалиста на periods2/3/4, 360 секунд на целую invocation. Временные caps включают host preparation, fit-only selection, prediction и свежий refit каждого компонента на supplied fit. Отдельные компоненты выполнялись последовательно с фиксированными долями исходного deadline. Все quality-вызовы завершились до deadline; увеличения cap, изменения seed/весов и технических model retries не было.

Calibration сохраняет legacy alignment: internal predictions при best iteration, свежий refit при `max(50, best)`; recheck-фитов нет. Фактические несовпадения: LGB period3 49→50, period4 31→50; numeric-only CB period4 39→50. Это объявленная часть алгоритма, а не скрытое изменение калибровки. Ансамбль явно согласует original-iloc calibration positions до history selection; обе модели после этого обучаются заново. Period2 имеет только один supplied-fit период: фиксированный fallback 400 деревьев, без внутреннего selection fit.

## Screen: внешний period4, cap 180 s

Fit 22,054 строк; held 12,946 строк. Меньше RMSE лучше. Полные рецепты находятся в `round5_regression_recipes.json`; source и все ablations сохранены в evidence.

| Arm | RMSE | Δ к I | MAE | SSE | Fits | Wall, s |
| --- | --- | --- | --- | --- | --- | --- |
| I | 46.824893 | +0.0000% | 24.866786 | 28385019.15 | 2 | 7.484 |
| LGB | 45.605542 | -2.6041% | 24.713852 | 26925937.89 | 2 | 7.584 |
| E25 | 46.097727 | -1.5529% | 24.517449 | 27510255.88 | 4 | 11.051 |
| E50 | 45.647485 | -2.5145% | 24.375438 | 26975487.92 | 4 | 11.508 |
| E75 | 45.482391 | -2.8671% | 24.445957 | 26780715.25 | 4 | 11.254 |
| N | 51.601956 | +10.2020% | 28.191051 | 34472114.98 | 2 | 7.479 |
| N25 | 47.129468 | +0.6505% | 25.191414 | 28755483.18 | 4 | 10.416 |
| N50 | 48.052998 | +2.6228% | 25.930890 | 29893487.16 | 4 | 10.785 |

Promoted **E75 и LGB**. Разница LGB/E50 меньше 0.2% RMSE, поэтому фиксированное правило практической ничьей предпочло более дешёвый LGB с двумя fits. N/N25/N50 хуже I; ни одна девятая конфигурация не добавлялась.

## Confirmation: cap 360 s, свежие fits

| Period | Arm | Fit rows | Held rows | RMSE | Δ к I | MAE | SSE | Fits | Wall, s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | I | 2117 | 9058 | 49.752709 | +0.0000% | 28.643148 | 22421557.65 | 1 | 3.860 |
| 2 | E75 | 2117 | 9058 | 50.535761 | +1.5739% | 30.855413 | 23132891.99 | 2 | 4.954 |
| 2 | LGB | 2117 | 9058 | 52.352502 | +5.2254% | 33.012242 | 24826026.11 | 1 | 3.731 |
| 3 | I | 11175 | 10879 | 55.794171 | +0.0000% | 30.312217 | 33866213.15 | 2 | 5.952 |
| 3 | E75 | 11175 | 10879 | 55.523756 | -0.4847% | 30.480059 | 33538733.12 | 4 | 8.182 |
| 3 | LGB | 11175 | 10879 | 55.812422 | +0.0327% | 30.764710 | 33888372.81 | 2 | 5.704 |
| 4 | I | 22054 | 12946 | 46.824893 | +0.0000% | 24.866786 | 28385019.15 | 2 | 7.628 |
| 4 | E75 | 22054 | 12946 | 45.482391 | -2.8671% | 24.445957 | 26780715.25 | 4 | 11.514 |
| 4 | LGB | — | — | не запускался | stop rule | — | — | 0 | 0 |

E75: I SSE **84,672,789.95**, candidate SSE **83,452,340.37**, два практических RMSE выигрыша; каждый fold-loss меньше 5%. Это не отменяет сегментный veto. У LGB period3 RMSE — практическая ничья (+0.0327%), но его State_1-сегмент нарушает порог. Поэтому две разные неудачные проверки — period2 и period3, и period4 не запускался. Screen period4/180 не подставлялся вместо отсутствующего confirmation/360. Неполная LGB серия не имеет pooled quality claim.

## Существенные сегментные потери

Порог зафиксирован до fits: число строк ≥ max(100, 1% held), I SSE ≥ 1% fold SSE; для крупных категорий и novelty partitions блокирует ухудшение >10% SSE с абсолютным приростом >1% контрольной fold SSE.

| Arm | Period | Segment | Rows | I SSE | Candidate SSE | Δ SSE | Absolute Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E75 | 2 | category:building_class/level:Residential | 6449 | 10123578.07 | 12006794.63 | +18.6023% | +1883216.56 |
| LGB | 2 | category:State_Factor/level:State_6 | 8364 | 18846634.36 | 21404037.78 | +13.5695% | +2557403.42 |
| LGB | 2 | category:building_class/level:Residential | 6449 | 10123578.07 | 13464180.91 | +32.9982% | +3340602.84 |
| LGB | 2 | category_novelty/common | 8709 | 17387383.00 | 20591607.14 | +18.4284% | +3204224.14 |
| LGB | 3 | category:State_Factor/level:State_1 | 1242 | 4703398.69 | 5301981.76 | +12.7266% | +598583.07 |

Категория `common` — часть novelty partition, не новая категория. Значимых veto на строках с новыми категориями не обнаружено; это не свидетельство прохождения точного cold gate. Пересекающиеся partitions нельзя суммировать. Все сегменты (включая небольшие группы с rows и absolute SSE, missing counts, fit-only target bands, novelty и крупные категории) сохранены для всех 16 quality invocations в `segments.json` архива. Полные base/mean/calibrated/component predictions также сохранены.

## History и границы оценки

| Period | Outer fit rows | Host INNER-fit rows | Discovery | Frozen schema |
| --- | --- | --- | --- | --- |
| 2 | 2117 | 0 | no_rows | None |
| 3 | 11175 | 2117 | single_period | None |
| 4 | 22054 | 11175 | proposer_limited | None |

Discovery выполнен ровно один раз на разрешённом host INNER-fit каждого перехода, 8M/48, `extended=False`. Schema=None у всех arms; точные cold/repeat/history группы **not assessable / unknown**. `schema_unavailable` не означает all-cold. Счётчик no_history_rows из неизменённого legacy incumbent не используется как доказательство cold membership. Base, mean и calibrated predictions совпадают во всех 16 quality invocations; положительного результата с историей нет. В LGB JSON `unresolved_configurations` означает неполную confirmation-серию, а не найденные разные схемы.

Known Year5 не оценивался и не использовался для выбора. Full TRAIN/исходный TEST нового алгоритма не запускались, поскольку победитель отсутствует. Отдельная post-freeze compatibility-проверка и native admission остаются задачами капитана только при будущем согласованном кандидате. Эти известные development-периоды не являются blind test или leaderboard score.

## Fits, стоимость и среда

**52 actual started / 52 completed physical fits**: screen 26 + confirmation 18 + flat smoke 8. Всего 16 quality invocations, 2 toy smoke invocations, 3 discovery операции. В ledger 20 записей: smoke агрегирован одной записью. Failed model invocations 0, model retries 0, API calls 0; GPU не применялся.

Наблюдаемая сумма wall time в общем ledger — **134.768947 s** (включая discovery, smoke, host preparation и diagnostics), лимит 10,800 s. Вложенные component/observer timers нельзя складывать: authoritative wall — `host_total_seconds`. Точные started/completed fit records содержат engine, phase, rows, время и component; строки/фазы внутреннего selection и supplied-fit refit сохранены отдельно.

Два CLI обращения (перед E25 и E50) остановились на повторно появившемся lock **до reservation и любого fit**. Точная длительность этих preflight failures не измерена; для общего расхода консервативно добавлено по 180 s, итого **494.768947 s**, меньше трёхчасового лимита. Исходный ledger не переписан. После проверки отсутствия активного runner, terminal status всех записей и SHA соответствующей summary был сохранён и удалён только подтверждённо устаревший lock; такие восстановления отмечены в supplementary evidence. Признаки согласуются с повторной материализацией файла средой, точная внешняя причина не доказана. Завершённые fits не повторялись; retry allowance не расходовался.

CPU8 задан в engines/environment. Linux Python3.12.14; numpy1.26.4, scipy1.12.0, pandas2.2.3, scikit-learn1.5.2, CatBoost1.2.8, LightGBM4.6.0, psutil6.1.1. Полный `pip freeze` приложен. Совпадение численных pins не является проверкой полного platform image.

Process inspection недоступна во всех 16 quality invocations. **Peak RSS, CPU seconds и process cleanup — unknown**. 8 GiB — конфигурационный предел, его соблюдение измерением не подтверждено. Полная native квалификация 32GB/10500s не заявляется. Стоимость отдельных исследовательских host-фаз и E75 component stages приведена в `round5_regression_integration_request.md`; реальная native INNER/outer/full-TRAIN стоимость не измерялась.

## Проверки и передача

12 model-free tests passed (0.045 s): strict-past/fit-only preprocessing, alignment, отсутствие mutation, frozen validation, shared deadline/fallback/fresh refit и gate/identity guards. Настоящий flat-package smoke вне checkout passed: два E50 invocation, 8 fits, 3.499797 s; пакет не требует каталога `agent/`, проверены row order и добавление экстремальной query-строки. Это toy contract test, не quality comparison. Независимый аудит пересчитал RMSE/MAE/SSE и ID для всех 68 таблиц predictions/ablations; hashes совпали. Windows-команды проверены по интерфейсу, но здесь не выполнялись.

- `round5_regression_results.json`: все raw summaries, source/input/split/recipe/dependency/output/receipt hashes, actual fits/times, environment, discovery, evaluation и accounting.
- `round5_regression_results.csv`: 20 ledger entries + 2 preflight blocks + явно не выполненный LGB period4. Не выполненная строка не считается model attempt.
- `round5_regression_recipes.json`: восемь полных измеренных frozen recipes и явное retain-incumbent решение.
- `round5_regression_reproduction.md`: Linux и Windows PowerShell; точный `.venv\Scripts\python.exe`, `C:\Windows\py.exe` при необходимости создания Python3.12 среды; существующая исправная среда сохраняется.
- `round5_regression_integration_request.md`: интерфейс, eligibility, fallback, measured supplied-fit costs и отсутствие запроса на активацию.
- `round5_regression_evidence.json`: SHA256, размер и manifest внешнего data-free ZIP. Архив включает summaries, predictions, source/recipes, receipts, fit ledgers, environment и operational supplement; исходные TRAIN/TEST, ответы и fitted models исключены.

Merge, runtime activation и платформенная отправка не выполнялись. Рекомендация капитану — сохранить текущий rescue.
