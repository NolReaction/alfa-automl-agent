# Round 5: итог интеграции classification OOF

**Итог: `incomplete`; default OFF.** Реализован явный `--classification-round4-mode {off,on}`, платный identity probe и ранний frozen OOF slot с обычным host lifecycle. Полный ON не измерен: два startup CatBoost timeout в OFF нарушили заранее запечатанный stop policy. OFF завершил свою ограниченную штатную работу и публикацию, после чего harness остановил серию. ON, дополнительные seeds, повторные большие fits и merge не выполнялись. Δ ON−OFF — `null`, инженерный порог +0.002 не оценён.

Тип планировавшегося сравнения: `model_plus_scheduling_plus_paid_probe`. Полученный OFF score — отдельная диагностика, не доказательство пользы OOF. Изолированный прирост Round 4 к нему не прибавляется. Public outer используется агентом для финального принятия кандидата и не является новым blind test; качество скрытого TEST неизвестно.

## Зафиксированный код и проверки

- Base: `66e9481cb61cb83cfecab53be6984d7d2bf829b3`.
- После native main наблюдался на `954ef923e69190c0ea1a4c8b379c0f711de37bdb`; исходная база сохранена, rebase не выполнялся.
- Native tested SHA: `4dce6ffae076ba40cfc659c8b81555c256d1cef3`; tree `ab3d9c5d0ca50a3d356b5ad3ea6c3c1ab6ce698c`.
- Run manifest SHA-256: `36a7edfc6be5a413e4a78baa1b41ce638c24c0ea1f899713cc868a26b7156cf7`.
- Неизменённый plan SHA-256: `44ea3d91908ec9d3ce8d9914cec22554865f6485204d94e4ad7768191dd1c635`.
- Flat ZIP SHA-256: `eb522aa656675d203bf69a2d0d223e7c00cd4466d44c2bc0fe02e471f45c4160`.
- Frozen OOF source: `1cb8a3f9df7588b5c4773351b9426c29eedd4c55f297f59fafeeaeefaf300862`.
- Frozen canonical OOF recipe: `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc`.

До native прошли **186 fit-free tests**, compileall и diff check. Отдельный настоящий smoke из flat ZIP вне checkout завершился за 3.373687 с: 400 synthetic fit rows / 12 predict rows, 2 started / 2 completed физических fit. Все 29 dependency origins принадлежат flat package. При host cap 120 с calibration reserve исчерпал допустимый calibration deadline; выполнен законный identity fallback и fresh final fit. Это contract smoke, не 10-fit OOF success и не native quality admission.

Во время native исходники, зависимости, guards и данные не менялись. После завершения затенение `identity` во внутреннем цикле аудитора исправлено локальным именем `inner_identity`. Отдельно исправлен неисполненный положительный qualification path: `promising_opt_in` теперь требует опубликованный OOF, как было указано в sealed plan; достигнутый OOF с ordinary winner получает `rejected`, сохраняя измеренную bundle Δ. Native runtime не изменён, текущий `incomplete` не зависит от обеих правок. Исходный аудитор сохранён по tested SHA, исправленный — отдельно. Все **30 native fit-free tests** после исправления проходят; новые тесты проверяют полную frozen identity после завершённого INNER и запрет promising-статуса при ordinary winner. Исправленный аудит дал идентичный OFF result. Original auditor SHA-256: `36f1236cde37b7484525eef968d95011c2e4d92491823a3ada511e23e9669f0c`; corrected auditor: `a26c94822c0e60f235dae01cb653d9de4a21fe17e8d9bc377700358bbcaa38d4`. Новых model runs при аудите не было. Final PR head содержит эту отдельную audit-only правку и итоговые документы; он не подменяет tested SHA.

## Фактический результат OFF

| Arm | Native / qualification | INNER Macro F1 | Published outer Macro F1 | OOF / probe |
| --- | --- | ---: | ---: | --- |
| OFF | completed / incomplete | 0.6482492770397070 | 0.6469928093378587 | 0 / 0 |
| ON | not started / incomplete | — | — | 0 / 0 |

Победитель — прежний `frequency_candidate_v1`: fit-only log-category counts для `proto`, `service`, `state` и равный ансамбль LGBMClassifier500 / CatBoostClassifier400. Полный actual recipe находится в results.json; это не frozen OOF LGB400/RF120. Published source hash `137c05075dec9f9044e8b7bf46c978235a3a182d9a31db1239eb3dd0b005f8ef`; recipe hash `859619b388cb801792b025c60366506615b9eb9c31098b06deb49a8f99b991ea`. Baseline outer — 0.6456188838250412; выбранный candidate outer — 0.6469928093378587. Оба outer CSV имеют receipts, выбор `candidate` — штатное outer acceptance.

Grouped seed42: outer fit70 155 / held17 515, INNER fit55 945 / held14 210, raw-group overlap0. Ordered classes, индексы, group fingerprints и input hashes сохранены в manifest/host-only evidence. Независимый stdlib rescore проверил точный порядок ID и повторил INNER и outer; полные confusion matrices и precision/recall находятся в JSON.

| Класс | INNER support | INNER F1 | Outer support | Outer F1 |
| --- | ---: | ---: | ---: | ---: |
| Analysis | 161 | 0.2269129288 | 192 | 0.2391752577 |
| Backdoor | 121 | 0.1192052980 | 192 | 0.1943127962 |
| DoS | 991 | 0.4679509632 | 1212 | 0.4766329598 |
| Exploits | 2739 | 0.6650602410 | 3296 | 0.6794988197 |
| Fuzzers | 1473 | 0.7277152419 | 1803 | 0.7447226313 |
| Generic | 3427 | 0.9905965325 | 4150 | 0.9885616938 |
| Normal | 4451 | 0.9121100700 | 5665 | 0.9168303864 |
| Reconnaissance | 746 | 0.8056338028 | 881 | 0.8000000000 |
| Shellcode | 93 | 0.6923076923 | 108 | 0.6560000000 |
| Worms | 8 | 0.8750000000 | 16 | 0.7741935484 |

## Стоимость и финализация

OFF: solver CLI budget3600 с, выделенный v3 budget3593.738795 с после подготовки, native elapsed2661.551 с; supervisor wall2663.881229 с. Подготовка manifest/package заняла 22.739552 с отдельно. Watchdog3630 с не расширял solver budget. Research остановлен по `finalization_reserve`; 54 ordinary local attempts (53 verified,1 failed), 3 existing extension attempts (3 verified). Из 54 ordinary — 16 warmup и38 adaptive. API attempts/responses0/0.

Полные host measurements выбранного кандидата, включая invocation overhead:

| Фаза | Fit rows | Predict rows | Host seconds |
| --- | ---: | ---: | ---: |
| INNER | 55945 | 14210 | 113.247156 |
| outer | 70155 | 17515 | 137.583821 |
| full TRAIN | 87670 | 41166 | 179.354262 |

Оба baseline/candidate прошли fresh full TRAIN refit; публикация связана с candidate outer/refit разными nonce и frozen source/recipe. Candidate full TRAIN87 670 → TEST41 166. Published `submission.csv`, full-refit output и last-valid checkpoint имеют один SHA-256: `cb368550f8a3709bb5dece5845fe692394b2f01f86e9e4c1920e542eb7de828f`. TEST проверен только по schema/ID/classes, labels не разыскивались.

| Категория | Уникальных invocations | Подтверждённый lower bound completed fit | Неизвестных invocations |
| --- | ---: | ---: | ---: |
| Startup | 19 | 19 | 2 |
| Ordinary, включая baseline lifecycle | 57 | 59 | 1 |
| Прежние extensions, включая winner lifecycle | 5 | 10 | 0 |
| Round5 probe / OOF | 0 | 0 | 0 |
| Native total | 81 | 88 | 3 |

78 invocations имеют completion evidence. **88 — source-inferred нижняя граница завершённых base-estimator fit calls**, а не event-verified total; точные actual started/completed остаются `null`. 592 path/hash bindings сохранены в `fit_accounting.json`; preprocessing fits, tree counts, pure prediction blending, smoke и fixtures не добавлены к native числам. Smoke2 fits учитываются отдельно.

Два уникальных startup отказа — `refine_5/selection` и `refine_6/selection`, timeout примерно81.277 и81.191 с, exit−9; повторные ссылки в JSON не удваивают их. У failed ordinary `adaptive_0038` (linear) отсутствует receipt, stdout/stderr пусты: причина и actual fits неизвестны, timeout не утверждается. ON не начат именно по sealed any-nested-timeout stop rule. OOF eligibility/support/admission/guards/feasibility не оценивались; native trained/accepted/applied rules0. Нулевые OOF counts не означают бесплатный guard rejection: его здесь не было.

## Среда, evidence и ограничения

Python3.12.14, Linux, cgroup8 CPU /8 GiB; 8 threads, GPU скрыта, API credentials удалены из solver environment. Полные pinned versions и source/dependency hashes — в исходном run manifest. Запрошенный supervisor sampled tree threshold6 GiB; прежние inner workers сообщают свой threshold28 GiB. Эти requested limits не являются наблюдённым RSS или hard OS limit. Process inventory недоступен: tree RSS, CPU и cleanup completeness — **unknown**. Linux result не квалифицирует Windows/32 GB или всю конкурсную платформу.

Ограниченный архив `round5-classification-host-evidence.zip` содержит original manifest/seal, полный tested source/flat package, plan/recipe/dependencies, native run.json, receipts, source/input/output hashes, split/group audit, completed INNER и published outer/TEST CSV, startup raw evidence, fit accounting, отдельный smoke, исходный и исправленный аудитор. Модели, full datasets, feature CSV и request objects исключены. Переносимый verifier использует только stdlib, повторяет score/receipt/source bindings и не может повысить `incomplete` до успешной пары. Проекция INNER labels подтверждена original TRAIN SHA до/после; без исключённого TRAIN архив не повторяет исходную feature-row equality проверку, выполненную host auditor.

Команды — в `round5_classification_host_reproduce.md`. MD/JSON/CSV согласованы; хеш самого архива передаётся отдельно в PR, чтобы не создавать самоссылку. **Гипотеза этого запуска закрыта с `incomplete`, default OFF; повторов, default ON, merge, платформенной отправки или принятого релиза нет.**
