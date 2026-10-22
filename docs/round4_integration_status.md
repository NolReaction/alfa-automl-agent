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
| ИИ-1 | work/round4-regression-v1 | Regression modules, новые признаки и fit-only history/base | Задание выдано, новый PR ещё не принят |
| ИИ-2 | work/round4-classification-v1 | Probability adapter, OOF/decision/features | Задание выдано, новый PR ещё не принят |
| ИИ-3 | work/round4-llm-v1 | Компактный INNER context и полезные Qwen trials | Задание выдано; host hook требует отдельного согласования |
| Капитан | work/round4-integration-v1 | Budget allocation, native3600, интеграция/выпуск | План зафиксирован; работа начата |
| Мурад | work/round4-murad-passport-v1 | Standalone экспорт existing feature profile | Задание выдано; production не затрагивает |

Наш [заранее записанный план](round4_integration_plan.md) отделяет конкретный allocator fix от неподтверждённых гипотез scheduler/cache. Ни одна новая модель или context policy пока не объявлена принятой. Первый длинный контроль — full-native classification3600. Итоги и точные проверенные SHA добавляются после завершения, а не заранее.
