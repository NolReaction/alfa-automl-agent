# Раунд 6: готовность фактического host INNER к замороженному OOF — результат

Публичная классификация (UNSW-NB15, `data/classification`), канонический путь host v3 на временной копии задачи.
Модели не обучались, identity probe не запускался, API и платформа не вызывались. Подробности:
`round6_inner_readiness_summary.md`, полная таблица support — `round6_inner_readiness_support.csv`,
все поля и hashes — `round6_inner_readiness_report.json`. План до измерения — `round6_inner_readiness_plan.md`.

## Краткий ответ

1. **Точный fit.** Host попробует OOF на `gate.research_train.iloc[gate.fit_rows]` объекта `ResearchWorkspace`,
   построенного `prepare_research` (`agent/research_v3.py:265`). Цепочка: TRAIN 87 670 строк → авто outer
   `grouped` (fit 70 155, holdout 17 515) → INNER `grouped` (fit **55 945**, held 14 210). Это не полный TRAIN
   и не outer fit. Тот же fit PR #51 передаёт в preflight (`:169-171`), identity probe (`:360`) и OOF (`:397`).
   Сверка с таблицей PR #51 и с её копией через `fit.csv`, как её читает кандидат: признаки, строки и порядок,
   метки и ID совпадают. 42 признака, 34 718 raw-групп.
   `row_hashes_sha256 = 9819ca9ed56ff21af833a12f444dda50fa9132cbb4599a804ceef761d455ecda`,
   `fit_csv_sha256 = 2439cb9da16e342bdc0e457778d206643172e031faba7019a9cc69f5e52b67f8`.
   Recipe `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc` совпадает с замороженным.
2. **Первый OOF fit.** Структурно начнётся: `structurally_compatible`, все gates
   `fit_oof_guarded_rule` пройдены, первый callback (fit fold 1) достигнут без обучения. Последний gate перед
   ним — класс в B ≥ 5 (`agent/oof_classification.py:319`), и он проходит впритык. Все структурные предусловия
   PR #51 (`grouped` outer и INNER, классы, роли, лимиты строк и ячеек для INNER, outer и refit) выполнены.
   Фактический старт ещё зависит от runtime: включён adaptive search, есть измеренный baseline, резерв на probe,
   свободный CatBoost-слот, identity probe завершился за ≤ 120 с и прошёл допуск по стоимости. Это `unknown`.
3. **Минимальный запас.** B / Worms: 5 строк (5 групп) при пороге 5 — запас **0**. Следующие: fit fold 1 / Worms
   21 строка при пороге 1 (запас 20), A / Worms 36 при пороге 5 (запас 31). Если в B окажется хотя бы на одну
   строку Worms меньше, gate `selection_insufficient_class_support` вернёт identity-правило; preflight PR #51 по
   тому же порогу (`:133`) пропустит OOF до probe.
4. **Стоимость подготовки до identity probe.** Модель-free часть — **80,1 с** (контрольный повтор 79,4 с, content
   совпал): TaskSpec/data load 3,0 с, outer workspace 30,0 с (из них outer `split_rows` 27,6 с), INNER split
   21,7 с, coverage на INNER fit 25,4 с. Ещё 1,3 с — запись и чтение `fit.csv` в каждом вызове кандидата.
   Конструктор `ResearchWorkspace` (≈ 54,7 с) host строит в любом прогоне v3. Собственная добавка OOF до probe —
   coverage в preflight: 25,4 с в этом процессе при лимите PR #51 60 с на отдельный процесс вместе с joblib и
   запуском. Пик памяти процесса за оба прогона — 555 МБ (`psutil peak_wset`, только этот процесс).
5. **Неизвестно до обучения.** Прирост Macro F1 на B ≥ 0,01 и падение F1 класса ≤ 0,02; время identity probe и
   допуск `4·p·1,5 + max(120, 2·p)`; время OOF-fit и refit; успешный свежий полный refit; RSS дерева процессов
   ≤ 6 ГиБ; стартовый baseline host; время subprocess preflight PR #51; приём платформой. Структурная
   совместимость не обещает прироста Macro F1.

## Поддержка по частям (строки / группы, запас)

| Часть | Порог | Worms | Shellcode | Backdoor | Analysis |
| --- | --- | --- | --- | --- | --- |
| allowed (INNER fit) | групп ≥ 2 | 41 / 41 (39) | 366 / 359 (357) | 560 / 529 (527) | 647 / 576 (574) |
| A (44 800 строк, ≥ 100) | строк ≥ 5 | 36 / 36 (31) | 287 / 282 (282) | 454 / 432 (449) | 524 / 464 (519) |
| B (11 145 строк, ≥ 60) | строк ≥ 5 | **5 / 5 (0)** | 79 / 77 (74) | 106 / 97 (101) | 123 / 112 (118) |
| fold 1 fit | строк ≥ 1 | 21 / 21 (20) | 192 / 188 (191) | 319 / 302 (318) | 360 / 320 (359) |
| fold 2 fit | строк ≥ 1 | 26 / 26 (25) | 192 / 188 (191) | 298 / 284 (297) | 331 / 292 (330) |
| fold 3 fit | строк ≥ 1 | 25 / 25 (24) | 190 / 188 (189) | 291 / 278 (290) | 357 / 316 (356) |

Остальные классы и held-части folds — в CSV. Запасы остальных классов больше 100.

## Вложенные границы

Все проверки по фактическим позициям TRAIN пройдены. Outer fit и holdout делят TRAIN без пересечений и общих
raw-групп. `research_train` совпадает с TRAIN на позициях outer fit. INNER fit и held делят outer fit без общих
групп, ни одна строка не попала в outer holdout. Аудит получил ровно INNER fit (группы и метки сверены с TRAIN).
A/B делят INNER fit без общих групп. Три fold лежат внутри A, не делят группы, каждая строка A ровно один раз
в held, B ни в один fold не попадает. 947 групп с разными labels (9 516 строк) остаются целыми группами.

## Сравнение с раундом 5

PR #52 проверял outer grouped fit (70 155 строк) как приближение INNER и получил Worms в B = 8 (запас 3).
Фактический INNER меньше на одно grouped-разбиение: 55 945 строк, Worms в B = 5 (запас 0). Приближение
завышало запас на 3 строки.

## Как получено

- База `114fc2c18595898e0f6ec099e42795eec4af0be9` (`origin/main`, merge PR #52 с правками капитана).
- План до измерения: `ac76add680f6e7ead252213b7a09b43cfef1f2f5`.
- Измерено из clean commit `97302482a9f538426f31fdd0514134bbecbaab1e` (код и тесты); после него код и пороги
  не менялись. Head — коммит с этим документом и отчётами.
- PR #51 читался как источник (`origin/work/round5-oof-host-v1` = `25e5967`), не импортировался и не запускался.

Выполненные команды (Windows, Python 3.12.0 из `.venv` основного checkout):

    git fetch origin --prune
    git worktree add ..\alfa-round6-murad -b work/round6-murad-inner-readiness 114fc2c18595898e0f6ec099e42795eec4af0be9
    ..\alfa-automl-agent\.venv\Scripts\python.exe -m unittest tests.test_round6_inner_readiness tests.test_round5_data_readiness tests.test_round4_feature_passport tests.test_research_workspace
    ..\alfa-automl-agent\.venv\Scripts\python.exe experiment_round6_inner_readiness.py --task-md ..\alfa-automl-agent\data\classification\task.md --output-dir ..\round6-reports\classification --control-repeat

Тесты: 65 OK за 56 с (15 новых в `tests/test_round6_inner_readiness.py` и соседние round 4/5 и workspace).
План называл `-v` и только round 5/6; запущен расширенный набор без `-v`.

Стоимость: измерение — 174 с wall на две модель-free подготовки (85,5 с и 84,7 с), без моделей и сети.
Исходные файлы задачи побайтово те же до и после, копия совпала с ними по hashes, host-проверка копии
(`check_source`) прошла.

## Ограничения и риски

- Запас 0 у Worms в B: любое изменение состава TRAIN, порядка классов или правил разбиения может перевести
  первый fit в `ineligible`. Это свойство данных и канонических разбиений, пороги не снижались.
- Время — один компьютер (ноутбук Мурада), один прогон и один контрольный повтор.
- Ссылки на строки PR #51 даны для `25e5967`; черновик может измениться.
- `content_sha256 = 4295acdf251f4c4969d342112ed87d7bb89df67a26293c034811e486dcc0c2af`,
  `config_sha256 = ed77a2aa87affcf72d9ddc16fc54d64b222fd81a39faef69ac25087ddfa65af9`.
