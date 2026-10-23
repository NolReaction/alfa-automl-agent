# Раунд 4: текущее состояние интеграции

Дата старта: 23 октября 2026, МСК. Общая база команды: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`.

## Принятый выпуск предыдущего раунда

Main e62 включает PR37/38/39/40/41/42: temporal scheduler, regression candidate и default CLI rollout, API reliability и classification tooling. Классификационная модель и её поздняя очередь сохранены. CLI rescue auto включается для подходящей adaptive RMSE задачи с остатком >300 секунд; Python API остаётся bool/default False. Отключение новой модели — `--no-rescue-regression`.

Новая регрессия прошла парный native900 на immutable13c9: INNER33.794488868→32.163919848; известный outer44.569657210→39.471458256 (−11.4387% RMSE). Оба arms сделали fresh full refit 53 308→22 449; независимые audits прошли. Это известный development outer, не hidden TEST или новый конкурсный балл.

Итоговое дерево выпуска: `59024a4f470939f01b4ba66a2dd6b27bfbd5718f`. ZIP `alfa_solution_20261022.zip`, SHA256 `fdfc8e31fec4fe5ef00c4dc92b1441e8670aaf3306ba27c9ceff25fd886686f9`. Все восемь pre-merge push/PR workflows прошли; один Windows frontend timeout потребовал точечного rerun без изменений кода/порога. Native измерения — 8 CPU/8 GiB/900 секунд, API0; live Qwen и полный 32 GB/10 500-секундный режим этим выпуском здесь не квалифицированы.

Старые `integration_pending` и отсутствие CLI-флага в исторических rescue-документах описывают ранний этап. Их числа сохранены, текущий статус задаёт этот документ.

## Работа текущего раунда

| Владелец | Ветка | Результат/граница | Статус |
| --- | --- | --- | --- |
| ИИ-1 | work/round4-regression-v1 | Версионированные fit-only regression features/base | [PR44](https://github.com/NolReaction/alfa-automl-agent/pull/44); production activation требует evidence и native admission |
| ИИ-2 | work/round4-classification-v1 | Probability adapter, OOF/decision/features | Первичное ревью 9ec3462; качество и rollout оцениваются отдельно |
| ИИ-3 | work/round4-llm-v1 | Ограниченный INNER context и action-policy | [PR45](https://github.com/NolReaction/alfa-automl-agent/pull/45); только inactive helpers, blanket early-stop не рекомендован |
| Капитан | work/round4-integration-v1 | Budget allocation, native3600, интеграция/выпуск | Fix 8a88dc3, 68 focused tests; итоговая native/CI приёмка — [PR43](https://github.com/NolReaction/alfa-automl-agent/pull/43) |
| Мурад | work/round4-murad-passport-v1 | Standalone экспорт existing feature profile | Задание выдано; production не затрагивает |

Наш [заранее записанный план](round4_integration_plan.md) отделяет конкретный allocator fix от неподтверждённых гипотез scheduler/cache. Новая модель и context policy не включаются только по факту merge экспериментальных helpers. Первый длинный контроль — full-native classification3600 на immutable e62, парный кандидат — immutable 8a88dc3, единственное runtime-отличие `agent/research_budget.py`. Итоговая таблица фактических native метрик, аудитов, CI и release SHA публикуется после завершения в PR43 и evidence выпуска; исходный план не переписывается под результат.

## Материалы капитана

- [Разбор пяти прежних native запусков](round4_integration_budget_review.md): непересекающиеся интервалы времени, причины refit-budget отказов и отсутствие доказанных повторных fits.
- [Контракт Qwen после ревью actual PR45](round4_integration_llm_handoff.md): корректный retry hook и причины не включать blanket stop. Mock fixtures не доказывают полезность живого Qwen.
- [Один длинный Windows-прогон принятого релиза](round4_windows_qualification.md): точный SHA, flat ZIP, pinned runtime, 32 GB и 10 500 секунд. Здесь Windows/full-budget не выполнен; это воспроизводимая последовательность, не сертификат квалификации.

Независимое первичное ревью новых направлений выполнено на зафиксированных heads: regression be06058 (31 pure/mocked tests), classification 9ec3462 (79 focused tests), Qwen 29d7564 (18 pure/mocked tests плюс сверка 59 source hashes). Это не blanket approval последующих коммитов и не повтор всех ML-экспериментов исполнителей. Обновлённые evidence и heads проверяются отдельно перед слиянием.
