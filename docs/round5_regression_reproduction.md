# Round5: команды воспроизведения зафиксированного эксперимента

CLI ниже сверены с implementation, команды Windows в этом окружении не выполнялись. Не следует повторно запускать уже завершённую текущую серию: для воспроизведения нужен новый каталог результатов и собственный clean checkout.

| Назначение | SHA |
| --- | --- |
| Зафиксированная общая база | `66e9481cb61cb83cfecab53be6984d7d2bf829b3` |
| Реально проверенный локальный implementation | `c0ba45118c41dfa4b0ad0360f925874692dd3082` |
| Опубликованный эквивалент implementation | `41f0b09c766682515684d4838c6c91dfa95bb154` |
| Общее дерево двух implementation commits | `aab125446d87e843aef9e0ca4ad83ba4518dd31f` |

Метаданные commit различаются; дерево исходников идентично. Для нового воспроизведения используется опубликованный implementation SHA, а не движущийся main/HEAD ветки. Итоговый PR может содержать дополнительные commits с отчётами: они не заменяют measured implementation SHA.

В проверенном Linux runtime: `/workspace/scratch/b69ed5d5758a/round5-env/bin/python`. Исходная задача: `/workspace/scratch/b69ed5d5758a/alfa-round5-regression/data/regression/task.md`. Фактический smoke уже прошёл: два E50 вызова, восемь started/completed physical fits, примерно 3.50 секунды. Ниже показан порядок для новой серии, не команда повторного учёта тех же восьми fits.

Завершённая измеренная серия сохранила incumbent: screen выбрал E75 и LGB; confirmation выполнил восемь вызовов. E75 получил два RMSE выигрыша, но не прошёл значимый Residential-сегмент периода 2 (SSE +18.60%). LGB остановлен после потерь на сегментах периодов 2 и 3; его период 4 не запускался. Freeze и full refit не выполнялись. При воспроизведении финалисты и остановка снова определяются только сохранёнными evidence и фиксированными правилами.

## Общая последовательная очередь

Сохранить следующий вспомогательный скрипт как `run_round5_queue.py` **вне checkout**, рядом с будущими результатами. Он вызывает существующий harness без изменения recipes. Screen всегда содержит ровно восемь arm; confirmation использует максимум двух фактически выбранных финалистов. После двух материальных неудач на разных confirmation-периодах arm больше не вызывается. Ни одной внешней Year5-оценки здесь нет.

Этот orchestration-фрагмент предоставлен для воспроизведения; сама измеренная серия запускалась через те же функции/CLI harness. Он не является дополнительным алгоритмом или девятой конфигурацией.

```python
import json
from pathlib import Path
import subprocess
import sys

repo, output, compact_dir, archive = map(lambda x: Path(x).resolve(), sys.argv[1:5])
expected = "41f0b09c766682515684d4838c6c91dfa95bb154"
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert head == expected, (head, expected)
assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
sys.path.insert(0, str(repo))
import experiment_round5_regression as lab

# prepare and the reserved/registered smoke must already be complete.
lab.plan(output)
assert not (output / "queue.lock").exists(), "Inspect an existing queue lock; do not remove an active lock"
assert not (output / "runs").exists(), "Use a fresh series; do not rerun completed arms"
assert lab.read(output / "validation/flat_smoke/summary.json")["status"] == "completed"

lab.discover(output, 4)  # exactly once, 8M/48, extended=False, allowed host INNER-fit
for arm in lab.ARMS:
    row = lab.run(output, "screen", 4, arm)
    print(json.dumps({k: row.get(k) for k in ("stage", "period", "arm", "status", "quality_metric")}), flush=True)

promotion = lab.promote(output)
finalists = promotion["finalists"]
assert len(finalists) <= 2
if finalists:
    for period in lab.DEVELOPMENT:  # 2, 3, 4 only
        if period != 4:
            lab.discover(output, period)
        lab.run(output, "confirmation", period, "I")
        for arm in finalists:
            previous_failures = sum(lab._fold_failure(output, earlier, arm)
                                    for earlier in lab.DEVELOPMENT if earlier < period)
            if previous_failures >= 2:
                print(json.dumps({"arm": arm, "period": period, "status": "stopped_after_two_failures"}), flush=True)
                continue
            lab.run(output, "confirmation", period, arm)

decision = lab.evaluate(output)
if decision["recommended_arm"] != "I":
    lab.freeze(output)
    # Same frozen history_config, including None; one 360-second full TRAIN
    # refit and original unlabelled TEST. No separate Year5 validation.
    lab.full_refit(output)
else:
    print("retain-incumbent: no new-candidate full refit", flush=True)

# Keep source clean through both operations. Outputs must remain outside git.
lab.compact(output, compact_dir)
print(json.dumps(lab.export(output, archive)), flush=True)
print(json.dumps(decision), flush=True)
```

Model/contract failures that the harness records as terminal failed attempts remain in the evidence and get no score. An unexpected exception stops this queue; do not silently rerun it. No automatic technical retry is included. A documented technical retry, if independently warranted, is limited to the one global allowance and uses the existing `--technical-retry-reason` interface with exactly the original recipe/seed/cap.

## Linux / Bash

Начать в существующем repository. Пути `R5_TASK`, `R5_WORK`, `R5_PARENT` заменить на свои. Интерпретатор разрешается до перехода в новый worktree; исправная существующая среда не переустанавливается. Если `.venv` отсутствует, создать обычную Python3.12 venv; requirements устанавливать только в эту новую среду после перехода на fixed SHA.

```bash
set -euo pipefail
R5_NEW_ENV=0
if [ ! -f .venv/bin/python ]; then
  python3.12 -m venv .venv
  R5_NEW_ENV=1
fi
# Resolve the directory, not the executable symlink: preserve the venv.
R5_PYTHON="$(cd .venv/bin && pwd -P)/python"
# In the measured hosted environment, use this already validated alternative:
# R5_PYTHON=/workspace/scratch/b69ed5d5758a/round5-env/bin/python
R5_TASK="/absolute/path/to/regression/task.md"
R5_WORK="/absolute/path/to/alfa-round5-reproduction"
R5_PARENT="/absolute/path/to/round5-reproduction-evidence"
R5_OUT="$R5_PARENT/series"
R5_SMOKE="$R5_PARENT/flat-smoke"
R5_SHA=41f0b09c766682515684d4838c6c91dfa95bb154

git fetch origin
git -c core.autocrlf=false worktree add --detach "$R5_WORK" "$R5_SHA"
cd "$R5_WORK"
test "$(git rev-parse HEAD)" = "$R5_SHA"
test "$(git rev-parse 'HEAD^{tree}')" = aab125446d87e843aef9e0ca4ad83ba4518dd31f
test -z "$(git status --porcelain)"
if [ "$R5_NEW_ENV" = 1 ]; then
  "$R5_PYTHON" -m pip install -r requirements.txt
fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

"$R5_PYTHON" -c 'import sys,numpy,pandas,scipy,sklearn,catboost,lightgbm; assert sys.version_info[:2] == (3,12); assert (numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,catboost.__version__,lightgbm.__version__) == ("1.26.4","2.2.3","1.12.0","1.5.2","1.2.8","4.6.0"); print(sys.version)'
"$R5_PYTHON" - "$R5_TASK" <<'PY'
import hashlib,json,sys
from pathlib import Path
task=Path(sys.argv[1]).resolve()
record=json.loads(Path("docs/round5_regression_plan.json").read_text())
for name, expected in record["input_files"].items():
    path=task.parent/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
print("Original input bytes match preregistration")
PY
"$R5_PYTHON" -m unittest tests.test_round5_regression_candidate tests.test_round5_regression_harness -v
"$R5_PYTHON" experiment_round5_regression.py prepare --task "$R5_TASK" --period-column Year_Factor --size-column floor_area --output "$R5_OUT"
"$R5_PYTHON" experiment_round5_regression.py reserve-validation --output "$R5_OUT"
set +e
ROUND5_NATIVE_SMOKE_OUTPUT="$R5_SMOKE" "$R5_PYTHON" -m unittest tests.test_round5_regression_native -v
R5_SMOKE_EXIT=$?
set -e
# Register also a failed smoke, so attempted physical fits stay accounted.
"$R5_PYTHON" experiment_round5_regression.py register-validation --smoke-directory "$R5_SMOKE" --output "$R5_OUT"
test "$R5_SMOKE_EXIT" = 0
"$R5_PYTHON" "$R5_PARENT/run_round5_queue.py" "$R5_WORK" "$R5_OUT" "$R5_PARENT/compact" "$R5_PARENT/round5-regression-evidence.zip"
test -z "$(git status --porcelain)"
```

Если уже имеется корректный отдельный interpreter, используйте его абсолютный путь вместо создания `.venv`. Не исправляйте несовпадающие версии переустановкой рабочей среды без разбора причины: для точного воспроизведения можно выделить новую среду. Проверка hashes относится ко всем первоначальным входам, включая task.md, а не к метрикам Year5.

## Windows / PowerShell

Начать в существующем repository. `.venv\Scripts\python.exe` разрешается до смены worktree. Для отсутствующей среды используется полный `C:\Windows\py.exe`, поскольку короткий `py` ранее выбирал неверный executable. Выход с ненулевым кодом проверяется после каждой native команды.

```powershell
$ErrorActionPreference = 'Stop'
$R5NewEnv = $false
if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    & 'C:\Windows\py.exe' -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python3.12 venv creation failed' }
    $R5NewEnv = $true
}
$R5Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$R5Task = (Resolve-Path 'C:\alfa-data\regression\task.md').Path
$R5Work = 'C:\alfa-round5-reproduction'
$R5Parent = 'C:\round5-reproduction-evidence'
$R5Out = Join-Path $R5Parent 'series'
$R5Smoke = Join-Path $R5Parent 'flat-smoke'
$R5Sha = '41f0b09c766682515684d4838c6c91dfa95bb154'

git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
git -c core.autocrlf=false worktree add --detach $R5Work $R5Sha
if ($LASTEXITCODE -ne 0) { throw 'worktree creation failed' }
Set-Location $R5Work
if ((git rev-parse HEAD) -ne $R5Sha) { throw 'Wrong implementation SHA' }
if ((git rev-parse 'HEAD^{tree}') -ne 'aab125446d87e843aef9e0ca4ad83ba4518dd31f') { throw 'Wrong implementation tree' }
if (git status --porcelain) { throw 'Source must be clean before measurements' }
if ($R5NewEnv) {
    & $R5Python -m pip install -r .\requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'New venv dependency install failed' }
}
$env:OMP_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'

$R5Check = @'
import sys,numpy,pandas,scipy,sklearn,catboost,lightgbm
assert sys.version_info[:2] == (3,12)
assert (numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,catboost.__version__,lightgbm.__version__) == ('1.26.4','2.2.3','1.12.0','1.5.2','1.2.8','4.6.0')
print(sys.version)
'@
& $R5Python -c $R5Check
if ($LASTEXITCODE -ne 0) { throw 'Existing runtime differs or cannot import; inspect it before changing dependencies' }
$R5InputCheck = @'
import hashlib,json,sys
from pathlib import Path
task=Path(sys.argv[1]).resolve()
record=json.loads(Path('docs/round5_regression_plan.json').read_text())
for name,expected in record['input_files'].items():
    assert hashlib.sha256((task.parent/name).read_bytes()).hexdigest() == expected['sha256'], name
print('Original input bytes match preregistration')
'@
& $R5Python -c $R5InputCheck $R5Task
if ($LASTEXITCODE -ne 0) { throw 'Input hashes differ from preregistration' }
& $R5Python -m unittest tests.test_round5_regression_candidate tests.test_round5_regression_harness -v
if ($LASTEXITCODE -ne 0) { throw 'Model-free tests failed' }
& $R5Python .\experiment_round5_regression.py prepare --task $R5Task --period-column Year_Factor --size-column floor_area --output $R5Out
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
& $R5Python .\experiment_round5_regression.py reserve-validation --output $R5Out
if ($LASTEXITCODE -ne 0) { throw 'Smoke reservation failed' }
$env:ROUND5_NATIVE_SMOKE_OUTPUT = $R5Smoke
& $R5Python -m unittest tests.test_round5_regression_native -v
$R5SmokeExit = $LASTEXITCODE
Remove-Item Env:ROUND5_NATIVE_SMOKE_OUTPUT
# Register attempted fits even when the smoke failed; then stop on failure.
& $R5Python .\experiment_round5_regression.py register-validation --smoke-directory $R5Smoke --output $R5Out
$R5RegisterExit = $LASTEXITCODE
if ($R5SmokeExit -ne 0 -or $R5RegisterExit -ne 0) { throw 'Smoke failed; evidence preserved; do not start the screen' }
& $R5Python (Join-Path $R5Parent 'run_round5_queue.py') $R5Work $R5Out (Join-Path $R5Parent 'compact') (Join-Path $R5Parent 'round5-regression-evidence.zip')
if ($LASTEXITCODE -ne 0) { throw 'Queue stopped; inspect retained evidence before any continuation' }
if (git status --porcelain) { throw 'Source changed during the experiment' }
```

## Отдельные команды для проверяемого продолжения

Если серия уже начата, сначала читать её `attempts.json`, существующие summaries и `promotion.json`. Новые output directories не превращают уже потраченные fits в бесплатные probes. Нельзя повторять завершённые arms, начинать вторую независимую серию бюджета или удалять чужой/активный lock.

Стандартные вызовы harness для очередной уже разрешённой операции:

```powershell
& $R5Python .\experiment_round5_regression.py discover --period 4 --output $R5Out
& $R5Python .\experiment_round5_regression.py run --stage screen --period 4 --arm E25 --output $R5Out
& $R5Python .\experiment_round5_regression.py promote --output $R5Out
& $R5Python .\experiment_round5_regression.py run --stage confirmation --period 2 --arm I --output $R5Out
# Use only an arm actually listed in promotion.json, after the two-failure check.
& $R5Python .\experiment_round5_regression.py evaluate --output $R5Out
# Only when evaluation.json recommends a new arm:
& $R5Python .\experiment_round5_regression.py freeze --output $R5Out
& $R5Python .\experiment_round5_regression.py full-refit --output $R5Out
& $R5Python .\experiment_round5_regression.py compact --destination (Join-Path $R5Parent 'compact') --output $R5Out
& $R5Python .\experiment_round5_regression.py export --destination (Join-Path $R5Parent 'round5-regression-evidence.zip') --output $R5Out
```

Эти отдельные строки — справочник интерфейса, а не команда запускать E25 повторно или делать full refit при отрицательном решении. При `retain_incumbent` перейти к compact/export без freeze/full-refit. Не создавать external Year5 fold и не подставлять обнаруженную ранее Year5 schema; frozen None остаётся None.

В текущей hosted-среде наблюдалось повторное появление уже удалённого `queue.lock` после terminal invocation. Это операционная проблема файловой системы, не основание повторять обучение. При таком событии вручную проверить отсутствие активного worker, terminal status всех attempts, соответствие lock последней завершённой операции и SHA её summary; сохранить lock и журнал проверки отдельно. Никогда не добавлять безусловное удаление lock в приведённую очередь. Не менять implementation во время серии.

До завершения compact/export не записывать отчёт или results в tracked каталог `docs/`: это изменит source identity. Перенос разрешённых summary/report artifacts в итоговую ветку выполняется только после серии. Data-free ZIP содержит разрешённые predictions, source/recipes, receipts, fit ledgers и hashes; исходные TRAIN/TEST, ответы held, fitted models и private native logs не публикуются. ZIP является evidence archive, не конкурсной отправкой и не native admission.
