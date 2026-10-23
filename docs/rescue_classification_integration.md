# Подключение классификационного кандидата: integration_pending

Проверенный исходный маршрут: `d05572846369f3b1e9cfa33f4008dda028db4fe0`.
Этот документ описывает возможную экспериментальную проводку интегратору. Общие router/scheduler-файлы в этой ветке не изменены. По завершённым измерениям **сохранить incumbent; default-победителя нет**. Средний paired-прирост CB Sqrt −0.0010494249, LGB −0.0004284534; у обоих нарушены условия среднего улучшения, падения на срезе и F1 отдельных классов. Увеличение native-бюджета для этих рецептов не рекомендовано.

## Необязательный экспериментальный рецепт

Если интегратор позднее решит проверить саму проводку, единственный описанный здесь opt-in — `lightgbm_none`, лучший по среднему Macro F1 среди двух новых финалистов: **0.6175072865** против **0.6179357399** фиксированного incumbent. Это не рекомендация включения. Полные измерения и причины отказа — `docs/rescue_classification_report.md`, машиночитаемое решение — `data/experiments/rescue_classification_v2/selection_summary.json`.

Точный измеренный нормализованный JSON:

```json
{
  "version": 1,
  "family": "early_stopping_boosting_v1",
  "base_recipe": {
    "action": "lightgbm",
    "model_config": {
      "model": "LGBMClassifier",
      "parameters": {
        "n_estimators": 300, "num_leaves": 31, "max_depth": -1,
        "min_child_samples": 20, "learning_rate": 0.05,
        "reg_lambda": 1.0, "colsample_bytree": 1.0, "class_weight": null
      }
    },
    "feature_config": {
      "missing_indicators": false, "signed_log": false, "rare_min_count": 0,
      "date_columns": [], "text_columns": []
    }
  },
  "validation": {"strategy": "grouped"},
  "max_iterations": 2000,
  "stopping_rounds": 50
}
```

Recipe SHA-256 — `d8ec2733aa118cca65b74e32b16e1668a5c969062a4d6cbba1b9ef134caccaa4`; source SHA-256 — `6b515d9bcebc505db96e86fbd4f4a8f9589c5e1d4657c61ba1aa0b88722d3eeb`. Импортировать `candidate_source` и `make_boosting_recipe` из `agent.boosting_candidate`; вызвать `make_boosting_recipe(base_recipe, {"strategy": "grouped"}, max_iterations=2000, stopping_rounds=50)`. `candidate_source()` возвращает строку source для сохранения в собственный новый файл перед `freeze_candidate`; source вызывает существующий `execute_boosting_candidate` и пишет `boosting_evidence.json`. `n_estimators=300` в base recipe заменяется результатом внутреннего выбора, а не задаёт итоговое число деревьев. Production не должен импортировать экспериментальный стенд или читать его артефакты.

Отдельный full-TRAIN refit LGB завершён за 42.887199 с: 87 670 fit-строк, 41 166 unscored TEST-строк, 151 выбранная итерация, `fresh_full_refit=true`. Локальный CSV проверен, SHA-256 `7cbbb8752f5c7e7b92f56e7fc9bfc4559a5c04f7bc21ce52ec8c29684c6349f8`. Это подтверждает доставку; платформенной отправки и основания повысить статус рецепта нет.

Фабрика возвращает source с контрактом `fit_predict(X_fit, y_fit, X_predict, context)`. Source и JSON один раз проходят `freeze_candidate`. Во всех фазах используется тот же `FrozenCandidate` и те же `code_hash`/`recipe_hash`. Внутренний подбор количества деревьев повторяется только по текущему fit; затем модель и препроцессор заново обучаются на всех переданных fit-строках. Ветка алгоритма не зависит от `context["phase"]`, имени файла или имени задачи. Частичных fitted-состояний между фазами нет.

## Какой слот заменить

В текущем `_continue_locally` (`agent/research_v3.py`) вызывается `local_candidate_plan`, затем `run_deadline_continuation`. `after_warmup=extensions` достигается **только после исчерпания всей очереди warmup**; переносить туда ещё один дорогой эксперимент недостаточно.

Первые восемь элементов `local_candidate_plan` для классификации фиксированы до пропуска дубликатов:

| Позиция | Рецепт |
| --- | --- |
| 1 | RF: 120 деревьев, depth 18, leaf 2, `balanced_subsample` |
| 2 | ExtraTrees: 160 деревьев, leaf 2, `balanced` |
| 3 | ExtraTrees: 200 деревьев, leaf 1, `balanced` |
| 4 | CatBoost: 100 итераций, depth 6, без весов |
| 5 | CatBoost: 100 итераций, depth 8, `SqrtBalanced`, индикаторы пропусков/редкие категории |
| 6 | LightGBM: 100 деревьев, 31 лист, без весов |
| 7 | LightGBM: 100 деревьев, 63 листа, `balanced` |
| 8 | LogisticRegression: 500 итераций, без весов, signed-log/индикаторы пропусков |

Далее идут три раунда refinement по семействам, затем до трёх изменений признаков. Порядок семейств refinement зависит от INNER-ranking. При пустом ranking глубокий CatBoost — `local_21`; полагаться на этот ID нельзя.

**Предлагаемая замена одного конкретного слота:** CatBoost refinement с `iterations=400`, `depth=10`, `learning_rate=0.03`. Остальные параметры и признаки этого контрольного слота наследуются из выбранной INNER-модели CatBoost. Найти слот по этой семантике в уже построенной очереди, вынуть его и поставить первым после измеренного baseline, до остальных warmup-рецептов. Включённый challenger занимает его место; исходный глубокий CatBoost второй раз не запускается. Общее число слотов и исходный общий deadline не увеличиваются.

Для корректного paired-сравнения **обе arms** должны уже содержать одинаковое перемещение этого слота. В `off` в раннем слоте исполняется вынутый обычный CatBoost; в `on` — измеренный классификационный рецепт. Поэтому сравнение `off/on` на новом общем SHA отвечает на вопрос о замене модели, а сравнение исходного main со всей новой системой остаётся отдельным опытом. Если контрольный слот отсутствует или уже учтён в `startup_seen`, записать явный skip в обеих arms; не создавать добавочный trial в обход дедупликации. Такой запуск не подтверждает достижение challenger.

Текущий `run_deadline_continuation` принимает один общий `code_hash`, а `evaluate` замыкает `BASELINE_SOURCE`. Нельзя подложить custom JSON под этот source или продолжать считать его ordinary-рецептом. Интегратору нужен ограниченный способ передать **source + recipe + source hash** раннего слота, сохранить его дедупликацию по `canonical_recipe_key(recipe, actual_code_hash)` и учесть его в существующем общем счётчике trials. Custom-слот не следует отправлять в `AdaptiveSearch.observe` как предложение TPE.

## Допуск, бюджет и отказ

Предлагаемые консервативные границы первой интеграции, одинаковые для обеих arms:

- только `classification`, `Macro F1`, `prediction_kind="labels"`, `adaptive_search=True`, открытая стадия `inner_research` и разрешённая task semantics;
- уже измеренный и неизменный baseline; существующие INNER fit/held; `gate.choice` и `gate.inner_split` имеют strategy `grouped`;
- точные feature-роли; ID/target исключены из групп и признаков; все объявленные классы есть в fit; 2–20 классов, минимум две различные группы признаков на класс; невозможное внутреннее grouped-разбиение — skip;
- первоначальный admission: fit ≤100 000 строк, predict ≤500 000 строк, ≤512 выходных признаков, fit/predict-матрица ≤10 000 000 ячеек каждая. Эти осторожные границы — предложение интегратору, не новые подтверждённые пределы модели;
- дополнительно сохранить штатные проверки `execute_boosting_candidate`: соответствие схемы входов, классов и метрики; все классы должны присутствовать в обеих частях внутреннего split. Не обходить их при допуске раннего слота;
- CPU, максимум 8 потоков, официальные версии пакетов. На нашей машине запрошен tree-RSS cap 6 GiB, но монитор недоступен из-за namespace: он не применён, peak RSS и cleanup равны null. Фактический общий hard cgroup — 8 GiB; не выдавать это за испытание на 32 GB. На машине интегратора проверять `resources.available`, а не только настроенный cap;
- один trial = внутреннее обучение с ранней остановкой плюс новый full-fit refit: максимум два обучения. Начальный cap раннего слота — 300 секунд для обеих arms, дополнительно ограниченный `state.research_timeout(300)` и исходным абсолютным deadline. Максимум 2 000 итераций, patience 50; внутренний код резервирует время на refit;
- `_assess_candidate` продолжает решать допуск по host INNER Macro F1 и измеренной стоимости финализации. Чтобы передать cap, достаточно предложенного необязательного параметра `maximum=1800`, передаваемого в уже существующий `state.measure_inner(..., maximum=maximum)`; только ранний слот задаёт `maximum=300`. Это ещё не внесённое изменение, существующие вызовы сохраняют прежний default;
- после INNER никаких специальных коротких путей: штатные freeze, outer acceptance, полный refit и публикация. Их timeout определяет существующий finalization reserve исходного бюджета. Не начинать новый бюджет после окончания старого;
- missing package/class, невозможное grouped-разбиение, cap, timeout, нехватка refit reserve или ошибка модели оставляют baseline/предыдущий принятый вариант. Все причины записываются. Нарушение целостности source/input не глотать как необязательную ошибку модели.

Схема предлагаемой проводки, **псевдокод, не готовый patch**:

```python
# Общий для off/on scheduler: после baseline INNER, перед остальной warmup.
control_slot = remove_named_depth10_catboost_refinement(queue)
if control_slot_is_present_and_not_already_measured and eligible_from_fit_only:
    source, recipe = (ordinary_source, control_slot) if mode == "off" else experimental_lgb()
    source_path = write_source_to_fresh_file(source)
    frozen = freeze_candidate(source_path, recipe, fresh_directory)
    feedback = _assess_candidate(state, frozen, hypothesis, origin="local", maximum=300)
    # Учесть в том же лимите trials, сохранить feedback/hash/skip; очередь без control_slot.
# Обычная guarded continuation и штатная финализация продолжаются с оставшимся бюджетом.
```

Имя предлагаемой настройки — `classification_rescue_mode: "off" | "on"`, default `off`. Сейчас такой CLI-флаг/настройка в main **не существует**. Интегратор должен явно добавить её и сохранять в `quality_options`; нельзя симулировать переключение сменой source SHA, скрытым monkeypatch или фазой обучения.

## Какие доказательства вернуть после подключения

Текущие результаты не требуют нового native-запуска или повышения бюджета. Если интегратор отдельно выберет экспериментальную проверку проводки, он возвращает общий integration SHA и проверенную команду с новой настройкой. Обязательное первое сравнение: последовательно `off`, затем `on`, одинаковые входы/хеши/split/8 CPU/RAM и бюджет **900 секунд**, порядок закреплён до запуска. По имеющимся результатам более длинный опыт не рекомендован. Только новые подтверждённые данные, снимающие провал условий качества и указывающие конкретную неопределённость бюджета, могут обосновать **одно** последующее сравнение на 1 800 или 3 600 с; достижение кандидата само по себе недостаточно.

Сохранить для каждой фазы:

- host `receipt.json`: `version=3`, `nonce`, `phase`, `code_hash`, `recipe_hash`, `output_hash`; source/recipe hashes должны соответствовать подключённому кандидату;
- модельный receipt: family/action, `status`, `fresh_full_refit`, `full_fit_rows`, `predict_rows`, class order, внутренний split с нулевым group overlap, выбранные/фактические итерации и причина остановки;
- host feedback: `verified`, `accepted`, `metric`, `reason`, `timing.feasible_estimate`; execution timeout, sampled peak RSS, доступность монитора и cleanup;
- `evaluations.final_check`, `candidate_acceptance`, `published_recipe`, `candidate_receipts.outer/refit`, `refit_execution`, `publication.new_result_published`; hash новой итоговой CSV равен `output_hash` полного refit;
- реальные `run.json.api.attempts/responses`, оба равны нулю. В изолированном стенде нули имеют происхождение `structural_no_provider_path`, а не являются прочитанными счётчиками native агента;
- сохранённые предсказания и независимый пересчёт Macro F1, precision/recall/F1/support каждого класса и confusion matrix. Отсутствие diagnostic receipt при timeout означает «достижение не подтверждено», а не доказательство отсутствия обучения.

До этих проверок статус остаётся `integration_pending`. Прирост локального Macro F1 не переводится в обещание платформенных баллов.
