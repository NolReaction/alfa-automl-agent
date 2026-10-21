# Итог ночной работы №1

Ветка `work/night-quality-v1`, [PR #14](https://github.com/NolReaction/alfa-automl-agent/pull/14). Добавлена явная политика `adaptive_v1` для offline AutoML и локального Research v3. **Default `legacy` сохранён**: независимая confirmation не прошла заранее заданный допуск. Подробности, таблицы и воспроизведение — в [отчёте](night_quality_report.md); все пары — в [JSON](night_quality_results.json).

## Замороженные версии

- Baseline: `390829239bfb245e70d866c02d4cd69d51d4c0d9`; runtime SHA-256 `423227637b7a0441e09d61954092dac8fb5069af5e4a4b2143f929af0d054b68`.
- Измеренный candidate: `ac628cfcf4fc9d87573c226dbc393b52f4e66c73`; runtime SHA-256 `50b802838b615bd23aeb509e062de961874e6cc71af3fd3dfa627b8c603c7ceb`.
- Последующий commit добавляет только harness, тесты и результаты; production runtime остаётся тем же. Проверена одна версия политики, после confirmation параметры не менялись.

Протокол зафиксирован до первого нового fit. Измерения выполнены из отдельных detached worktrees. Входы, splits, audit answers, sources, harness и версии библиотек запечатаны до запусков. Один fit за раз, не более 8 потоков. Реальных API requests и платформенных отправок: 0.

## Завершённые серии

| Серия | Runs | W/T/L | Solver wall B/C, s |
|---|---:|---|---:|
| Development, 90 s, 9 families × 2 seeds | 36 | 8/6/4 | 387.957 / 339.600 |
| Confirmation, 180 s, 6 families × 2 новых seeds | 24 | 6/2/4 | 260.791 / 225.051 |
| Research v3 API failure, 180 s | 4 | 2/0/0 | 62.406 / 63.339 |
| Public TRAIN sanity, 180 s | 8 | 3/0/1 | 384.975 / 309.304 |

Все **72/72** завершены штатно с корректным CSV, неизменными inputs/source и полным refit по отдельному v2 re-audit. Solver timeouts/failures: 0. Telemetry RAM/CPU на этом host недоступна; wall times не выдаются за peak RSS или CPU measurements.

Confirmation: signed MAE median gain −4.7734% хуже допуска −3%; практически значимый ≥5% выигрыш только на heavy-tailed MAE (+11.1132%), нужны две families. Public regression period 4 ухудшилась на 23.42%: консервативный reserve 202.378 s при остатке 175.403 s отклонил все новые попытки. Этот случай и точный механизм сохранены. Public Macro F1 вырос на 0.011975 и 0.022922, но public train уже исследовался раньше и не заменяет независимую confirmation. Трёхчасовые поиски и private score не измерялись.

Под игнорируемым `data/experiments/night_quality/` находятся:

- `dev_ac628_frozen`, `confirm_ac628_frozen`, `research_failure_ac628_frozen`, `public_ac628_logged`: оригинальные запечатанные plans/reports и результаты;
- отдельные `*_reaudit_v2.json`, не меняющие старые v1 flags;
- `all_summary.md`, соответствующий сохранённому JSON.

`dev_ac628` был первоначальным preflight без fits и заменён immutable plan. Более ранняя попытка public preflight не создала plan и не обучала модели. Выполненные plans нельзя исполнять повторно: replay запрещён.

## Проверки и целостность

Финальный локальный `dev_check.py`: **667 tests, 436.295 s, OK (5 OS/resource-related skips)**. Все 87 профильных night-quality tests, включая настоящий запуск из временного flat ZIP, прошли. Исходные ID, ранний prior, обученный итог, полные inputs/config/durations/checkpoint и независимая метрика проверены.

CI измеренного runtime `ac628cfc`, run `39525138920`: Ubuntu 645 core / 227 app, Windows 645 core / 227 app, все launcher/ZIP steps успешны. Штатный workflow повторяет проверки итогового head; актуальные нативные результаты — в [PR checks](https://github.com/NolReaction/alfa-automl-agent/pull/14/checks).

У candidate development trial 1 пуст stdout полного seed worker. Исторический v1 flag остаётся false. Отдельный v2 подтвердил завершение по trusted source/launcher/config/full inputs/durations/checkpoint и пересчитал score. Причина пустого stdout не установлена. V1 helper не изменён ни на один байт. Новый auditor SHA-256 `b5f95610d69cfce9217fd65230c40441635951d7aa213e05371ebe21c4cc8ec2`.

Файлы параллельной работы №2, общий benchmark, recovery, packager, приложение и зависимости не менялись. Main сохраняется для общего review двух работ. Отправленный исходный `delivery/solution.zip` неизменен, SHA-256 `6493f721b2d7a742f7ee89dd1dc92e0a48939fac9b56b2f6b6b96820a1da3026`.
