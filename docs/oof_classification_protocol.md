# OOF Macro F1: протокол v1

База до реализации и опытов: `664bd5a38c4ba8e1451163917bbdff93979b4b54`.
PR #16 принят; проверки обоих обязательных предков (`6e7b5c019a438c3e5187ef6c96107bb329cb5678`
и `7d54bd6d9179cf57e4cfc4b39d3762092fcdd0aa`) завершились с кодом 0.
Ветка: `work/oof-macro-f1-v1`. Изменяются только шесть файлов из SECOND_AI_OOF.md.

## Гипотеза и границы

Это продолжение существующего guarded Macro F1. Старый пилот дал 3 улучшения
и 6 ухудшений; guarded follow-up — 2 улучшения и 7 identity. Перенос на полный
refit тогда не проверялся. Эти результаты не пересчитываются как новое подтверждение.
Новые seeds: development 131, 379; confirmation 2203, 4001, 10009.
Семейства известны, новые реализации не являются неизвестными типами задач.

Только метки Macro F1 меняются через `argmax(p * multipliers)`. Исходные
вероятности сохраняются; LogLoss/AUC этим правилом не калибруются.
Модуль не подключается к Research v3, общим defaults или CLI.

## Протокол до первого audit

1. Внешние роли: allowed train 1500 строк, outer audit 400, production audit
   400. В duplicate-семействе группы из четырёх строк не пересекают роли.
2. Host до fit фиксирует A/B примерно 70/30 через GroupShuffleSplit(seed+31),
   затем три GroupKFold только внутри A. Ответы не выбирают эти разбиения.
   Никаких повторных seeds/splits после нехватки класса.
3. Три модели дают OOF-A. Четвёртая обучается на A и предсказывает B. Ни одна
   calibration-модель и ни один её preprocessing не получает ответы B.
4. Существующий `fit_guarded_decision_rule` сначала выбирает кандидата на A,
   затем B только допускает его или identity. Пороги и grid не меняются.
5. JSON-правило и его хеш фиксируются перед outer fit. Пятая модель учится на
   A+B и предсказывает outer. Правило не переобучается.
6. То же правило применяется после production-refit на allowed+outer (1900
   строк) к отдельным 400 production audit. Ответы production audit не участвуют
   ни в одном fit. Это полный refit относительно выделенного development train,
   а не обучение на всех 2300 строках с последующей оценкой на них же.

Categorical shift здесь означает IID/domain shift без временного признака,
а не замену временной задачи random OOF. `split_policy='time'` явно не поддержан
и возвращает identity. Невозможный host split регистрируется отдельно; неверные
структурные индексы и вероятности являются ошибкой контракта.

## Модель, сравнения и стоимость

Одна заранее выбранная модель: RandomForestClassifier, 64 trees, max_depth=10,
min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample',
n_jobs=8, random_state=seed случая. Numeric median imputer и categorical
most_frequent + OneHotEncoder(handle_unknown='ignore') обучаются заново
только на X_fit. Ни early stopping, ни tuning модели по B/audit нет.

В OOF-плече identity и правило оцениваются на одной финальной модели и одних
вероятностях. Отдельное контрольное плечо использует ту же фиксированную
модель без calibration. Оба плеча имеют одинаковый общий wall-time cap,
включающий fit, prediction, calibration, serialization и production-refit:
180 s synthetic / 900 s public. Порядок identity/OOF чередуется по case index.
Фиксированный контроль не расходует оставшееся время на AutoML search;
сравнение не доказывает преимущество над полноценным AutoML при том же времени.

OOF-плечо: 3 OOF + A→B + allowed→outer = 5 fit, затем отдельно production-refit
(всего 6). Identity: 1 allowed→outer, затем production-refit (всего 2).
Чистая дополнительная цена calibration — 4 fit. При identity из-за нехватки
поддержки fit могут быть пропущены; фактические counts фиксируются.

Synthetic calibration deadline = исходный arm deadline − 60 s резерва,
каждый calibration subprocess ограничен 25 s. Для public резерв 300 s,
calibration subprocess 120 s. Финальные fit используют остаток того же arm
deadline. Контроллер не перезапускает завершённые пары и не меняет cap по audit.
Все fit последовательны, не больше 8 threads; тяжёлые suites одновременно
с серией не запускаются. Через `agent.execution.run_python` действует timeout
и существующая очистка дерева. Передача timeout callback сама не останавливает
обычную Python-функцию. Настроенный memory monitor cap 8 GiB не равен
измеренному RSS и не является жёстким OS/container memory limit.

## План, артефакты и повторный audit

`--prepare-only --output-dir NEW` до обучения сохраняет model/options, все
генерации, индексы, поддержки, версии, SHA кода и hashes source/input/split.
`--run-prepared DIR` запускается ровно один раз. `--re-audit DIR` читает
сохранённые вероятности/правила и пересчитывает F1 без нового обучения.
Изменение runtime-исходников или окружения после prepare отклоняется.
Большие данные, модели, requests, receipts и predictions остаются только в
игнорируемом `data/experiments/oof_classification/`.

Каждый дочерний fit получает только X_fit/y_fit/X_predict. Preprocessing
statistics/categories и hashes запроса/модели/вероятностей сохраняются для
проверки. Это граница передаваемых данных, не OS-песочница для недоверенного
кода под тем же пользователем. Правило хешируется перед refit и проверяется
после обеих фаз. JSON-wrapper не содержит огромной OOF-матрицы.

При наличии `data/classification/task.md` используются известные public TRAIN
grouped folds 0,1,2, не independent confirmation. При отсутствии данных
пишется `skipped_missing_data`; замена из сети не скачивается. Полный public
production-refit может дать вероятности test, но без его ответов метрика
не заявляется. Synthetic production audit выделен заранее отдельно.

API calls 0; `.env` не читается; platform submissions, ZIP и merge отсутствуют.
