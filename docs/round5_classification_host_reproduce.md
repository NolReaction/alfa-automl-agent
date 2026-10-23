# Round 5: воспроизведение

**Статус: smoke пройден; native-пара остановлена с `incomplete`.** OFF опубликован и независимо проверен; два startup timeout запрещают ON. Δ null, default OFF. Команды ниже документируют интерфейс и исходный план; уже начатую серию повторять нельзя. Исходная база `66e9481cb61cb83cfecab53be6984d7d2bf829b3`, ветка `work/round5-oof-host-v1`; default всегда OFF.

Код, полный `docs/round5_classification_host_recipe.json` и preregistered plan коммитятся до первого большого fit. `prepare` требует чистый checkout и записывает фактический integration SHA/tree, plan/package/source/dependency/input hashes в ignored `run_manifest.json` + `manifest_seal.json`. Самоссылки SHA внутри плана нет. Обе arms используют один snapshot; documentation-only evidence commit после пары не меняет tested runtime.

## Linux: одна управляемая пара

Ниже сохранён интерфейс исходного запуска. Он не является разрешением повторить остановленную пару. Для иной отдельно разрешённой серии нужны новый preregistered plan, clean commit и новые directories. Из корня `/workspace/scratch/df1d37ea75a8/alfa-round5-oof-host`:

```bash
classification_python=/workspace/scratch/df1d37ea75a8/alfa-automl-agent/.venv/bin/python
"$classification_python" experiment_round5_classification_native.py prepare \
  --output data/round5-oof \
  --public-task /workspace/scratch/df1d37ea75a8/alfa-automl-agent/data/classification/task.md \
  --workspace /workspace/scratch/df1d37ea75a8/round5-native-runtime \
  --python "$classification_python"
"$classification_python" experiment_round5_classification_native.py smoke --output data/round5-oof
"$classification_python" experiment_round5_classification_native.py run --output data/round5-oof
"$classification_python" experiment_round5_classification_native.py audit --output data/round5-oof
```

Каждая следующая команда разрешена только после успешной предыдущей. `prepare` не обучает модели: запечатывает sources, пять public файлов, штатные split/group fingerprints и один `flat.zip`, извлекает пакет вне checkout, создаёт отдельные свежие `off/task/` и `on/task/`. Подготовка имеет проверяемый cap 600 с и отдельный elapsed. Python harness обязан совпадать с `--python`; путь к venv не заменяется реальным target symlink.

`smoke` — отдельный single-use contract check импортов/fresh fit/predictions из flat package: synthetic fit 400 / predict 12, worker cap 120 с, supervisor 150 с. Actual fits сохраняются отдельно. Из-за полного calibration reserve такой короткий smoke не доказывает применение OOF rule и не является native quality admission.

`run` удерживает `/tmp/alfa-research-lab-cpu-0.lock` на всей паре, выполняет **OFF → аудит OFF → ON**, записывает `pair_started.json` и точные argv. При invalid/incomplete OFF, source/input/receipt drift, timeout, memory stop или `cleanup.complete=false` ON не запускается. Автоматического retry/resume нет; `stop.json` сохраняет причину. Повторный `run` на начатой серии запрещён. `audit` пересчитывает сохранённые CSV без обучения.

Фактические solver argv внутри harness имеют следующий вид; это справка, **их не нужно повторно запускать после `run`**:

```bash
"$classification_python" /workspace/scratch/df1d37ea75a8/round5-native-runtime/package/solution.py \
  /workspace/scratch/df1d37ea75a8/round5-native-runtime/off/task/task.md \
  --mode offline --research-search adaptive --search-profile legacy --time-budget 3600 \
  --classification-round4-mode off
"$classification_python" /workspace/scratch/df1d37ea75a8/round5-native-runtime/package/solution.py \
  /workspace/scratch/df1d37ea75a8/round5-native-runtime/on/task/task.md \
  --mode offline --research-search adaptive --search-profile legacy --time-budget 3600 \
  --classification-round4-mode on
```

Harness задаёт task directory как cwd. Ручной вызов solver не заменяет lock, supervisor, последовательную проверку и manifest audit. Общий native budget ≤7 200 с; ON включает preflight ≤60 с, один paid probe ≤120 с и OOF ≤600 с на достигнутую фазу. Эти расходы не добавляются к 3 600 с. Ordinary CB удаляется только после фактической OOF-попытки; skip после платного probe сохраняет очередь, но уменьшает remaining trial cap.

## Среда и границы ресурсов

Выбранный Python проверен как **3.12.14**. `prepare` фиксирует все dependency versions и повторно сравнивает среду перед arms; installs/fetch/rebase/edits между ними запрещены. Текущий Linux cgroup прочитан напрямую: `cpu.max=800000 100000`, `memory.max=8589934592` — 8 CPU / 8 GiB. Это не Windows/32 GB qualification.

Штатный sanitized environment скрывает GPU, удаляет API credentials и задаёт 8 threads для OMP/OpenBLAS/MKL/NumExpr/Veclib. Одинаковый requested sampled tree memory threshold **6 GiB**, прежний supervisor. Native watchdog **3 630 с** включает 30 с startup/cleanup allowance без продления solver budget; trusted cleanup использует discovery 1 с, drain 2 с, root wait 5 с. Actual worker/supervisor/cleanup wall записываются раздельно. При недоступном monitor RSS/CPU/cleanup остаются unknown; successful publication может иметь `provisional_unknown_telemetry`. Child self RSS не является peak всего дерева.

## PowerShell: документированный интерфейс

**Windows execution ещё не проверено.** Сначала разрешить абсолютный executable до перехода в worktree. При необходимости создать Python 3.12 venv через `C:\Windows\py.exe`; короткий `py` ранее выбирал неверный executable. Пример с подготовленным окружением и короткими новыми roots:

```powershell
$classificationPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$classificationCheckout = 'C:\work\alfa-round5-oof-host'
Set-Location $classificationCheckout
& $classificationPython experiment_round5_classification_native.py prepare `
  --output data\round5-oof `
  --public-task C:\classification\task.md `
  --workspace C:\r5runtime `
  --python $classificationPython
if ($LASTEXITCODE -ne 0) { throw 'prepare failed; do not run' }
& $classificationPython experiment_round5_classification_native.py smoke --output data\round5-oof
if ($LASTEXITCODE -ne 0) { throw 'smoke failed; native forbidden' }
& $classificationPython experiment_round5_classification_native.py run --output data\round5-oof
if ($LASTEXITCODE -ne 0) { throw 'native pair stopped; no retry' }
& $classificationPython experiment_round5_classification_native.py audit --output data\round5-oof
```

Для создания venv отдельная команда: `& C:\Windows\py.exe -3.12 -m venv .venv`; после этого используется точный `.venv\Scripts\python.exe`. Public task root должен содержать те же пять файлов/hashes, что в плане; Windows lock path берётся из manifest, Linux-путь не переносится буквально.

## Evidence и решение

Фактические точки входа: `run_manifest.json`, `frozen_identity.json`, `off/report.json`, `stop.json`, итоговый `report.json`. `on/report.json` отсутствует, поскольку ON не запускался; ниже — `run.json`, execution/cleanup, receipts, split/group audit, child events и independently rescored outer CSV. Full TRAIN/TEST delivery подтверждается только если lifecycle реально дошёл до нового verified full refit; TEST labels отсутствуют, качество неизвестно.

Тип сравнения `model_plus_scheduling_plus_paid_probe`. `promising_opt_in` требует ON−OFF ≥.002 и опубликованного OOF; иначе reached OOF получает `rejected`, отсутствие достигнутого OOF — `not_reached`, незавершённая/невалидная пара — `incomplete`. Ordinary winner в ON относится к bundle, не к доказанной пользе OOF. Результат и bounded raw archive исключают модели/исходные большие datasets. Merge, default ON, конкурсный релиз и платформенная отправка не выполняются. Полный протокол — `docs/round5_classification_host_plan.md`, контракт — `docs/round5_classification_host_integration.md`.


## Проверка сохранённого архива без обучения

Распаковать `round5-classification-host-evidence.zip` в новый короткий путь. Из каталога распакованного evidence:

```bash
python3 portable_verify.py . > independent_verification.json
```

PowerShell (здесь не проверен):

```powershell
& C:\Windows\py.exe -3.12 .\portable_verify.py . | Set-Content -Encoding utf8 .\independent_verification.json
```

Verifier использует stdlib, повторяет 57 completed INNER CSV и published outer score, проверяет TEST format/ID/classes, source/recipe/receipt bindings, отдельный smoke и deterministic fit accounting. Он сохраняет `incomplete`, Δ null и отсутствие OOF quality claim. Архив не содержит full datasets/models/request objects; проверка исходных feature rows выполнена original host auditor и отдельно corrected auditor с теми же source/package/input hashes. INNER label projection доверяет SHA-verified original TRAIN provenance, поскольку TRAIN исключён.

Original auditor сохранён в `source_tested/experiment_round5_classification_native.py`; исправленный — в `postrun_audit/experiment_round5_classification_native.py`; их reports/hashes/provenance — в `evidence/postrun_audit/`. Direct `audit --output data/round5-oof` с текущего postrun HEAD намеренно отклоняет старый source seal: native guard не ослаблялся. Для переданного архива использовать portable verifier, а не изменять manifest и не повторять обучение.
