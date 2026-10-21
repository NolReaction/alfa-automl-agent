# NIGHT 02: состояние и передача

Ветка `work/night-unseen-v1`, PR https://github.com/NolReaction/alfa-automl-agent/pull/15 .
База `390829239bfb245e70d866c02d4cd69d51d4c0d9`.
Подробный протокол, дефекты и команды: `docs/night_unseen_report.md`.

## Завершено

- Зафиксирован каталог 38 случаев: 19 valid, 3 difficult, 10 negative,
  3 metamorphic и 3 scale/budget. Метаморфные источники указаны явно.
- Исправлены 12 групп подтверждённых дефектов своей runtime-зоны: строгие CSV,
  вещественные значения, literal labels/categories, объявления метрик,
  probability mapping и standalone validation. Модельный поиск и recovery не менялись.
- Основная пара: по 13 полных solver результатов, все 9 метрик и 10 expected
  refusals у базы и candidate. Re-audit подтвердил все CSV/refit receipts.
- Отдельный независимый пересчёт формулами подтвердил 26 результатов и
  312 хешей артефактов, совпадение метрик до 1e-12 без раскрытия таблиц качества.
- Default CLI без флагов: по 3 full result у базы и candidate, API settings отсутствуют.
- Дополнительные схемы, literal Markdown labels, группы/время и перестановки
  вероятностей проверены реальным solver. Два valid difficult split ограничены
  возможностями валидации и остаются solver_error, не full result.
- Известный линейный metamorphic case candidate упал при 30 s; отдельный
  ограниченный повтор 60 s дал full result. Первый error сохранён в матрице.
- По 2 малых scale-probes у обеих версий дали полный refit. Два намеренных
  8-секундных прерывания skipped до запуска из-за недоступного PID inventory.
- Исправлены новые Windows-тесты и остановка потомков независимого runner.
  Теперь сохраняются PID/create_time, а forced cleanup без подтверждения
  прерывает серию. Это best effort polling, не OS sandbox.

## Точные версии

- `461d8c97261f7bf41ea016775e233dc32a29ab84`: runtime fixes, core/default qualification.
- `78db10f489d5f759fa67d7aafd2a2947dd107278`: только переносимость тестов.
- `3281a863f1f45b92b78309db9eee36bc7a457fa0`: runner cleanup и его тесты,
  продолжение оставшихся случаев. Все шесть runtime-файлов, solution.py и
  генераторы побайтно совпадают с `461d8c9`.

Seed 81221 — публичная qualification/development матрица. Закрытая
confirmation-серия и чужой model-policy candidate здесь не оценивались.
Численные audit quality-таблицы/targets остаются в игнорируемых результатах.

## Проверки и блокер до merge

89 профильных tests и последующие 32 tests стенда прошли; последний набор
имеет один пропуск detached-child proof из-за PID namespace. App: 227 tests,
9 skips, OK. Обязательный dev_check: 646 tests, 1 error, 5 skips.
Native Linux/Windows на 78db: по 646 tests, только тот же V2 error.
Native CI 3281a863: по 651 tests на Linux/Windows, только тот же V2 error.
Linux: 2 skips, Windows: 1 skip. Реальный detached-worker cleanup proof прошёл
в Linux CI; на Windows POSIX-only proof skipped. Подробные job links в отчёте.

Единственная общая ошибка: legacy V2 после порчи исходного train скрывает
integrity ValueError за step-limit RuntimeError. Это несовместимость строгой
валидации с V2 recovery, введённая этой веткой. Владелец agent/research.py
должен вызвать state.check_source() вне catch выбора, до recover. Точный diff
и существующий reproducer приведены в отчёте. Чужой файл не правился,
валидация ради зелёного CI не ослаблялась. Автоматического merge нет.

## Результаты и восстановление

`data/experiments/night_unseen/` содержит manifest/summary, raw predictions,
refit-артефакты, независимые checks и re-audit. Первая extra-серия оборвана
перезапуском exec-service после трёх записанных результатов; у четвёртой
попытки terminal exit неизвестен. Это явно записано в stage1_interruption.json.
Сохранённые результаты не переобучались; оставшиеся выполнены в новых каталогах,
stage2-index.json завершён. Нельзя повторно запускать уже начатый output dir.

Проверки завершены, исходы и ограничения зафиксированы; новых обучений нет.
Контрольные точки и проверенные Git bundles находятся в
`data/experiments/night_unseen/checkpoints/`. Перед возобновлением читать
checkpoints/resume.json и завершённые summaries; не повторять всю матрицу.
Реальных конкурсных API-запросов, platform submissions и merge: 0.
Отправленный delivery/solution.zip не перезаписывался.
