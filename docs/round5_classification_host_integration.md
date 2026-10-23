# Round 5: контракт host-интеграции

**Статус: реализовано, default OFF; native-пара завершена как `incomplete`.** Документ сверён с helper, CLI/dispatcher, host и harness на базе `66e9481cb61cb83cfecab53be6984d7d2bf829b3`. До native прошли 186 fit-free tests и отдельный flat-package smoke (2 fresh fits); чистый SHA/tree и manifest запечатаны. OFF опубликовал проверенный результат, но два startup timeout остановили серию до ON. Полный native OOF и его польза не измерены. Итоговые evidence — в `round5_classification_host_results.md` и JSON/CSV.

Round 5 заменяет предложение старого `round4_classification_integration.md` о переносе CB в обеих arms: **OFF сохраняет прежний порядок полностью**. Только ON добавляет раннюю замену и оплаченный probe. Тип сравнения — `model_plus_scheduling_plus_paid_probe`; прирост не приписывается одной калибровке. Принятый regression rescue-v1 сохраняется.

## Вход и идентичность

Реализован CLI `--classification-round4-mode {off,on}`, default `off`; Python `run_agent` и `run_agent_v3` также default off. `agent/research.py` только валидирует/передаёт параметр через прежний dispatcher; OFF сохраняет прежние forwarding kwargs. CLI отклоняет ON вне поддержанного v3 adaptive route. Requested mode сохраняется в `quality_options.classification_round4_mode`; `classification_round4` содержит отдельные `requested_mode`, `reached_mode`, `status`, `reason`. До фактической OOF-попытки `reached_mode="off"`, даже если probe уже обучился.

Существующая factory, без изменения Round 4 runtime:

```python
from agent.round4_classification_candidate import make_recipe, checked_recipe, candidate_source
from agent.round4_classification_probability import fixed_incumbent_recipe

recipe = checked_recipe(make_recipe(fixed_incumbent_recipe(), classes, variant="oof"))
source = candidate_source()
```

`classes` — точный ordered TaskSpec, не runtime dataset switch. Полный JSON в `docs/round5_classification_host_recipe.json` создан и проверен этой factory без fits; canonical hash совпал с авторским `cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc`. Hash standalone source — `1cb8a3f9df7588b5c4773351b9426c29eedd4c55f297f59fafeeaeefaf300862`. Mismatch блокирует native. `dependency_snapshot` сохраняет source hashes перечисленных `DEPENDENCIES` и package versions; `verify_dependencies` вызывается через `V3State.check_source`. Файлы host/candidate/OOF/guard/models/features/splits/supervisor и frozen candidates перепроверяются. INNER/outer/full refit используют одну frozen identity; внутренний fit-only guard заново выбирает правило или identity на разрешённом fit.

## Host routing

`_assess_candidate(..., maximum=1800)` передаёт отдельный cap в `V3State.measure_inner`, сохраняя прежний вызов без дополнительного kwarg при default 1800. OOF использует `maximum=600`; both-stage feasibility проверяет candidate outer/refit estimates также против этого maximum. Host INNER оценка и финальное принятие сохранены.

Точка интеграции — `_continue_locally` после построения `recipes`/`seen` и до ordinary continuation. Семантически находится ещё не исполненный CatBoost iterations 400/depth10/learning_rate .03. Это третий CatBoost refinement в `agent.search.refinement_plan`; `local_candidate_plan` перемежает refinement families по INNER-ranking, поэтому позиция/`local_N` не стабильны. На базе сначала идут ordinary exploration recipes, затем round-robin refinements и feature variants. OFF их не перемещает.

ON вызывает `agent.round5_classification_host.run_round5_host` до ordinary continuation. `semantic_slot` проверяет параметры и `canonical_recipe_key`; helper создаёт frozen identity/OOF source, а ordinary `evaluate` продолжает использовать `BASELINE_SOURCE`. При actual OOF attempt исходный CB удаляется из оставшейся очереди; при skip до attempt очередь сохраняется. Если probe уже потрачен, очередь сохраняет состав, но trial allowance уменьшается. Остальные ordinary slots сохраняют порядок. Успешный или неуспешный probe и OOF расходуют действующий cap; coordinator получает `64 − early_invocations` оставшихся ordinary/adaptive попыток, без добавочного TPE наблюдения custom.

До каждого этапа сохраняются source/input/baseline guards. Custom проходит `_track_local_candidate`, `_assess_candidate`, INNER selection/feasibility, terminal freeze, outer acceptance, fresh full refit/publication. Ошибки identity не должны поглощаться как безобидный optional skip. После freeze новые попытки запрещены.

## Admission и probe

`cheap_eligibility`: adaptive/open INNER, measured baseline, разрешённая semantics, classification Macro F1 labels и совместимая grouped policy. Rows/shape/roles/class-order/cell bounds для всех будущих фаз выполняются до export/copy/augmentation: fit 100 000, predict 500 000, features 512, matrix cells 10 000 000, classes 2–50. Затем `bounded_preflight` через `preflight_worker`/штатный `run_python` проверяет raw-feature group/A/B/fold support только на INNER-fit, cap **60 с** из текущего research allowance; staging и проверки входят в тот же 3 600-секундный бюджет. Preflight не обучает моделей. Support/guard не ослабляются: A/B ≥100/60, каждый класс ≥5, B gain ≥.01, class drop ≤.02, group overlap 0.

Один совместимый identity adapter cost-only probe выполняется только ON; уже seen identity без compatible cost receipt даёт skip. Перед ним должны быть доступны все **120 с** cap сверх finalization reserve, повторно после preflight. Helper вызывает `state.measure_inner(identity, maximum=120)` напрямую, не assessor: полученная метрика не меняет best и не считается quality search. Собственный identity recipe, отдельные source/recipe/input/receipt hashes, полный wall elapsed и actual fit events сохраняются, включая неуспех. Два fits утверждаются только для подтверждённого successful probe.

После probe/подготовки текущий research allowance пересчитывается. Допуск: `4*p*1.5 + max(120,2*p) <= min(600, available_research_seconds)`. Incomplete/timeout/incompatible probe, неверное покрытие, отсутствующий CB или недостаток времени дают явный skip. Уже потраченное время/attempt остаётся в accounting. Изолированные старые timings не заменяют текущий compatible probe.

## Budget и lifecycle evidence

OOF сохраняет invocation 600 с, callback 90 с, reserve calibration 120 с и return/check margin 5 с, исходный абсолютный deadline и штатный supervisor. В outer/full refit `stage_limit` уменьшает host allocation до `min(600, allocator_timeout)` только при точном frozen OOF source+recipe; `final_stage_budgets` сохраняет исходный `allocator_timeout_seconds` и фактический `timeout_seconds`. OFF allocations не меняются. До 10 физических fits на invocation; до 30 для трёх достигнутых фаз, плюс 2 у successful probe. Identity fallback после guard может стоить столько же, сколько accepted rule.

`V3State.record_timing` записывает полный `run_candidate` wall. `finalization_plan` оценивает candidate_outer и candidate_refit отдельно, включая measured OOF/copy/predict overhead, а `BudgetPolicy.finalization_timeouts` защищает baseline stages. Не заменять это временем последнего fit и не менять `research_budget.py` без отдельного доказанного запроса. Проверка both-stage feasibility обязательна; row scaling остаётся heuristic.

В каждом достигнутом этапе нужны code/recipe/nonce/phase/input/output identity, receipt, fit/predict rows, `fresh_full_refit`, split/group hashes, child event started/completed/cancelled, trained/accepted/applied и причины guard. Raw callback count не равен physical fit count. При неполных events неизвестные counts не превращаются в нули. Published winner подтверждается outer/refit receipts и совпадением final CSV hash; валидный emergency CSV не означает квалификацию OOF.

## Отчёт и фактически достигнутые gates

`classification_round4` хранит `manifest`, `preflight`, `probe`, `admission`, `oof`, `invocations`, `used_invocations`, `ordinary_remaining_trial_cap`, `ordinary_slot_removed`, `replaced_ordinary_key`. `invocation_evidence` собирает частичные child events; `verify_result` связывает успешный result с context/receipt/frozen identity и child artifacts. Отказ callback с допустимым fresh identity fallback должен отличаться от integrity mismatch; итоговая проверка этого пути обязательна до seal.

Фактически выполнена одна OFF arm из preregistered пары: native completed, outer Macro F1 0.6469928093378587 независимо подтверждён. Два startup CatBoost timeout делают qualification `incomplete`; ON не начат, Δ null, OOF/probe/rules trained/accepted/applied 0. OFF победитель — прежний frequency_candidate_v1, не новый OOF. Runtime не менялся; после остановки исправлены только audit-only затенение identity и qualification gate опубликованного OOF; добавлены два regression tests (30 native tests PASS), original/corrected audit дали один OFF result. Команды и ограничения — в `round5_classification_host_reproduce.md`; результаты — в MD/JSON/CSV. Повторных fits, default ON и merge нет.
