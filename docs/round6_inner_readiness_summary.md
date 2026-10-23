# Готовность host INNER к OOF: structurally_compatible

Фактический INNER fit канонического host v3 на временной копии задачи: авто outer, его research fit, INNER fit/held. Это не INNER-контекст для Qwen и не допуск модели. Модели не обучаются, identity probe не запускается, API не вызывается. Структурная готовность не означает, что OOF повысит Macro F1.

| Поле | Значение |
| --- | --- |
| scope | actual_host_inner_fit_only_oof_structure |
| Решение | structurally_compatible |
| Причины | — |
| Первый OOF fit (структурно) | starts — oof_support_gates |
| Предусловия PR #51 / runtime | passed / unknown |
| recipe_sha256 | `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc` (замороженный: да) |
| Outer | grouped: fit 70155, holdout 17515 |
| INNER | fit 55945, held 14210 |
| Входы не изменились | да |
| content_sha256 | `4295acdf251f4c4969d342112ed87d7bb89df67a26293c034811e486dcc0c2af` |

## Точный fit

Источник: PR #51 25e5967 agent/round5_classification_host.py:169-171 gate.research_train.iloc[gate.fit_rows]. Вид кандидата: agent/research_candidate.py:299-300 fit.csv; driver :214 dtype=str keep_default_na=False, :224.

| Поле | Значение |
| --- | --- |
| Строк / признаков | 55945 / 42 |
| Совпадение с таблицей PR #51: признаки / строки и порядок / метки / ID | да / да / да / да |
| row_hashes_sha256 | `9819ca9ed56ff21af833a12f444dda50fa9132cbb4599a804ceef761d455ecda` |
| fit_csv_sha256 | `2439cb9da16e342bdc0e457778d206643172e031faba7019a9cc69f5e52b67f8` |

## Вложенные границы

| Проверка | Статус | Счётчики |
| --- | --- | --- |
| outer_partition | passed | fit_rows=70155, holdout_rows=17515, shared_rows=0, uncovered_rows=0 |
| research_is_outer_fit | passed | rows=70155 |
| outer_groups_disjoint | passed | shared_groups=0, strategy=grouped |
| inner_partition | passed | fit_rows=55945, held_rows=14210, shared_rows=0, uncovered_rows=0 |
| inner_outside_outer_holdout | passed | fit_rows_in_holdout=0, held_rows_in_holdout=0 |
| inner_groups_disjoint | passed | shared_groups=0, strategy=grouped |
| audit_input_is_inner_fit | passed | rows=55945 |
| ab_partition | passed | A_rows=44800, B_rows=11145, shared_rows=0, uncovered_rows=0, shared_groups=0 |
| fold_1 | passed | fit_rows=30108, held_rows=14692, rows_outside_A=0, shared_rows=0, shared_groups=0 |
| fold_2 | passed | fit_rows=29693, held_rows=15107, rows_outside_A=0, shared_rows=0, shared_groups=0 |
| fold_3 | passed | fit_rows=29799, held_rows=15001, rows_outside_A=0, shared_rows=0, shared_groups=0 |
| folds_hold_A_once | passed | fold_count=3, A_rows_held_once=44800, B_rows_held=0 |

## Минимальный запас

Запас 0: B / Worms (rows ≥ 5, строк 5, групп 5)

## Поддержка классов: строки / группы (запас)

| Класс | allowed | A | B | fold_1_fit | fold_2_fit | fold_3_fit |
| --- | --- | --- | --- | --- | --- | --- |
| Analysis | 647 / 576 (574) | 524 / 464 (519) | 123 / 112 (118) | 360 / 320 (359) | 331 / 292 (330) | 357 / 316 (356) |
| Backdoor | 560 / 529 (527) | 454 / 432 (449) | 106 / 97 (101) | 319 / 302 (318) | 298 / 284 (297) | 291 / 278 (290) |
| DoS | 3929 / 1645 (1643) | 3150 / 1324 (3145) | 779 / 321 (774) | 2147 / 873 (2146) | 2054 / 889 (2053) | 2099 / 886 (2098) |
| Exploits | 10661 / 7047 (7045) | 8489 / 5622 (8484) | 2172 / 1425 (2167) | 5721 / 3732 (5720) | 5597 / 3756 (5596) | 5660 / 3756 (5659) |
| Fuzzers | 5816 / 5412 (5410) | 4668 / 4335 (4663) | 1148 / 1077 (1143) | 3120 / 2885 (3119) | 3071 / 2857 (3070) | 3145 / 2928 (3144) |
| Generic | 12423 / 2248 (2246) | 10086 / 1787 (10081) | 2337 / 461 (2332) | 6737 / 1219 (6736) | 6782 / 1169 (6781) | 6653 / 1186 (6652) |
| Normal | 17884 / 16899 (16897) | 14380 / 13575 (14375) | 3504 / 3324 (3499) | 9665 / 9093 (9664) | 9549 / 9075 (9548) | 9546 / 8982 (9545) |
| Reconnaissance | 3618 / 2637 (2635) | 2726 / 2084 (2721) | 892 / 553 (887) | 1826 / 1372 (1825) | 1793 / 1381 (1792) | 1833 / 1415 (1832) |
| Shellcode | 366 / 359 (357) | 287 / 282 (282) | 79 / 77 (74) | 192 / 188 (191) | 192 / 188 (191) | 190 / 188 (189) |
| Worms | 41 / 41 (39) | 36 / 36 (31) | 5 / 5 (0) | 21 / 21 (20) | 26 / 26 (25) | 25 / 25 (24) |

Пороги: allowed — групп ≥ 2; A, B — строк ≥ 5; fold fit — строк ≥ 1. Held-части folds — в CSV.

## Ограничения кандидата

| Ограничение | Порог | Наблюдение | Выполнено |
| --- | --- | --- | --- |
| fit_rows | max 100000 | 55945 | да |
| declared_classes | max 50 | 10 | да |
| A_rows | min 100 | 44800 | да |
| A_min_class_rows | min 5 | 36 | да |
| fold_fit_min_class_rows | min 1 | 21 | да |
| B_rows | min 60 | 11145 | да |
| B_min_class_rows | min 5 | 5 | да |

## Проверки audit_oof_fit

| Проверка | Статус | Причина |
| --- | --- | --- |
| contract | passed | — |
| grouped_split | passed | structured_grouped_split |
| ab_disjoint | passed | — |
| folds_disjoint | passed | — |
| held_cover_A_once | passed | — |
| oof_support_gates | passed | first_fit_reached_without_training |
| child_input_caps | passed | — |

Дубли raw-признаков: групп с копиями 3710, групп с разными labels 947 (строк 9516).

## Предусловия host из PR #51

| Условие | Статус | Источник |
| --- | --- | --- |
| macro_f1_labels | passed | PR #51 25e5967 agent/round5_classification_host.py:71-73 |
| task_semantics | passed | PR #51 25e5967 agent/round5_classification_host.py:74-75 |
| inner_available | passed | PR #51 25e5967 agent/round5_classification_host.py:78-79 |
| grouped_policy | passed | PR #51 25e5967 agent/round5_classification_host.py:80-82 |
| class_count_and_order | passed | PR #51 25e5967 agent/round5_classification_host.py:83-87 |
| feature_roles | passed | PR #51 25e5967 agent/round5_classification_host.py:88-95 |
| row_caps | passed | PR #51 25e5967 agent/round5_classification_host.py:97-101 |
| feature_matrix_caps | passed | PR #51 25e5967 agent/round5_classification_host.py:102-103 |
| adaptive_search_on | unknown | PR #51 25e5967 agent/round5_classification_host.py:68-69 |
| measured_baseline | unknown | PR #51 25e5967 agent/round5_classification_host.py:76-77 |
| probe_reserve | unknown | PR #51 25e5967 agent/round5_classification_host.py:104-105, :345-346 |
| ordinary_catboost_slot | unknown | PR #51 25e5967 agent/round5_classification_host.py:306-308 |
| identity_probe_completed | unknown | PR #51 25e5967 agent/round5_classification_host.py:359-376 |
| cost_admission | unknown | PR #51 25e5967 agent/round5_classification_host.py:377-383 |

## Время (с, эксклюзивно)

| Этап | Секунды |
| --- | --- |
| task_copy_and_hashes | 0.671 |
| task_spec_and_data_load | 2.965 |
| outer_workspace | 30.048 |
| inner_extraction | 23.053 |
| fit_table_equivalence | 0.73 |
| coverage | 25.396 |
| guard_cap_audit | 0.963 |
| boundary_checks | 1.475 |
| source_checks | 0.055 |

Модель-free подготовка host до identity probe: 80.114 с (TaskSpec/data load + ResearchWorkspace (outer, INNER split, файлы workspace) + coverage на INNER fit; без стартового baseline, LLM, dependency_snapshot и subprocess preflight PR #51). Рендер экспорта: 0.003 с, запись трёх файлов на диск не входит.

Пик памяти: 555307008 байт (psutil peak_wset, scope this_process_whole_run).

Контрольный повтор: content совпал — да, подготовка host 79.421 с. Исходный результат сохранён.

## TEST

- load_task -> load_data открывает и проверяет test.csv (agent/task_spec.py:285-286, offline_baseline.py:128-145).
- load_task_data снова читает test.csv; data_fingerprint хеширует его (agent/research_workspace.py:292-294, :438).
- choose_validation для классификации смотрит только на дубли признаков TRAIN (agent/automl.py:158-163); для регрессии она сравнивает периоды TEST (agent/automl.py:164-182).
- Ограничения PR #51 по форме полного refit используют число строк TEST (PR #51 :97-103).
- audit_oof_fit не читает файлов и TEST: только переданные X_fit, y_fit, context.

## Не оценено без обучения

| Пункт | Требование | Статус | Почему |
| --- | --- | --- | --- |
| selection_macro_f1_gain | Δ Macro F1 на B ≥ 0.01 | unknown | нужны OOF-вероятности; аудит не обучает модели |
| selection_class_f1_drop | падение F1 любого класса на B ≤ 0.02 | unknown | нужны OOF-вероятности; аудит не обучает модели |
| time_admission | 4·probe·1.5 + max(120, 2·probe) ≤ 600 с (docs/round4_classification_integration.md) | unknown | нужен измеренный wall time identity probe на этом fit |
| fresh_full_refit | успешный финальный fit на всех строках fit | unknown | проверяется только обучением |
| final_predict_caps | ≤ 500000 predict-строк и ≤ 10000000 ячеек | unknown | predict-таблица host не входит в fit-only режим |
| memory | RSS дерева процессов ≤ 6442450944 байт | unknown | без обучения потребление памяти моделью не измеряется |
| identity_probe_seconds | identity probe ≤ 120 с (PR #51 :21, :360) | unknown | identity probe обучает модель и здесь не запускается |
| cost_admission | 4·probe·1.5 + max(120, 2·probe) ≤ доступного research-времени (PR #51 :377-383) | unknown | нужен измеренный wall time identity probe |
| startup_baseline | измеренный baseline до OOF (PR #51 :76-77) | unknown | стартовый поиск host обучает модели (agent/research_v3.py prepare_baseline) |
| pr51_preflight_subprocess | preflight PR #51 ≤ 60 с в отдельном процессе (PR #51 :22, :157-194) | not_evaluated | код PR #51 не запускается; та же coverage измерена в этом процессе |
| platform_admission | приём решения платформой | unknown | платформа не вызывается |

## Ограничения

- Группа — хеш всех raw-признаков строки, как в agent.splits.split_rows; копии строки с любыми labels остаются в одной группе.
- Время вложенных вызовов внутри конструктора ResearchWorkspace и audit_oof_fit измерено обёртками вызова без изменения кода; этапы эксклюзивны, их сумма не превышает общего времени.
- Время одного прогона на этом компьютере. Стартовый baseline-поиск host, LLM-диалог, dependency_snapshot и subprocess preflight PR #51 не входят: они обучают модели или запускают процессы.
- Память — пик этого процесса за весь прогон, если платформа его отдаёт; это не RSS дерева процессов кандидата.
