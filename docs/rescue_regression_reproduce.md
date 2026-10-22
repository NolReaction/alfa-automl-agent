# Воспроизведение regression rescue

Измеренный код: `b0485f70226461c69af0086511f00f4752d76e22`.
Нужен свежий каталог; одна тяжёлая очередь CPU. Команды ниже выполняют
изолированный скрининг/полный refit, а не отсутствующее native-подключение.

## Linux

```bash
set -e
git clone https://github.com/NolReaction/alfa-automl-agent.git ar-reg
cd ar-reg
git switch --detach b0485f70226461c69af0086511f00f4752d76e22
python3.12 -m venv ../ar-env
../ar-env/bin/python -m pip install -r requirements.txt -c requirements.platform.txt
git worktree add --detach ../ar-base d05572846369f3b1e9cfa33f4008dda028db4fe0
../ar-env/bin/python prepare_data.py regression
../ar-env/bin/python - <<'PY'
from pathlib import Path
from experiment_rescue_regression_screen import prepare, native, discover, diagnose, screen, freeze_decision, full_refit
r = Path('data/r')
prepare('data/regression/task.md', r, 'Year_Factor', 'floor_area')
for year in (4, 2, 3):
    native(r, '../ar-base', year, 900)
    discover(r, year)
    report = r / 'discovery' / f'y{year}' / 'extended' / 'discovery_report.json'
    diagnose(r, year, report)
    for engine in ('catboost', 'lightgbm'):
        screen(r, year, engine, 360, report)
# This is the already-declared decision, not a new automatic tuning step.
freeze_decision(r, 'catboost')
discover(r, 5)
native(r, '../ar-base', 5, 900)
report = r / 'discovery/y5/extended/discovery_report.json'
diagnose(r, 5, report)
screen(r, 5, 'catboost', 360, report)
full_refit(r, 360)
PY
```

## PowerShell: короткие пути и CPU

Используется конкретный Windows launcher `C:\Windows\py.exe`, чтобы не попасть
на посторонний `py` из system32. Рабочие каталоги должны быть свободны.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    function Run-Checked {
        $program = $args[0]
        $commandArgs = @($args | Select-Object -Skip 1)
        & $program @commandArgs
        if ($LASTEXITCODE -ne 0) { throw "$program failed: $LASTEXITCODE" }
    }
    foreach ($path in @('C:\ar-reg', 'C:\ar-env', 'C:\ar-base')) {
        if (Test-Path $path) { throw "Choose unused short directories: $path already exists" }
    }
    Run-Checked git clone https://github.com/NolReaction/alfa-automl-agent.git C:\ar-reg
    Set-Location C:\ar-reg
    Run-Checked git switch --detach b0485f70226461c69af0086511f00f4752d76e22
    Run-Checked 'C:\Windows\py.exe' -3.12 -m venv C:\ar-env
    $python = 'C:\ar-env\Scripts\python.exe'
    Run-Checked $python -m pip install -r requirements.txt -c requirements.platform.txt
    Run-Checked git worktree add --detach C:\ar-base d05572846369f3b1e9cfa33f4008dda028db4fe0
    Run-Checked $python -m unittest discover -s tests -p 'test_rescue_regression_*.py' -q
    Run-Checked $python prepare_data.py regression
    Run-Checked $python experiment_rescue_regression_screen.py prepare --task data/regression/task.md --output data/r --period-column Year_Factor --size-column floor_area
    foreach ($year in @(4, 2, 3)) {
        Run-Checked $python experiment_rescue_regression_screen.py native --output data/r --checkout C:\ar-base --years $year --budget 900
        Run-Checked $python experiment_rescue_regression_screen.py discover --output data/r --year $year
        $discovery = "data/r/discovery/y$year/extended/discovery_report.json"
        Run-Checked $python experiment_rescue_regression_screen.py diagnose --output data/r --year $year --discovery $discovery
        Run-Checked $python experiment_rescue_regression_screen.py screen --output data/r --year $year --engines catboost lightgbm --budget 360 --discovery $discovery
    }
    Run-Checked $python experiment_rescue_regression_screen.py freeze --output data/r --engine catboost
    Run-Checked $python experiment_rescue_regression_screen.py discover --output data/r --year 5
    Run-Checked $python experiment_rescue_regression_screen.py native --output data/r --checkout C:\ar-base --years 5 --budget 900
    $discovery = 'data/r/discovery/y5/extended/discovery_report.json'
    Run-Checked $python experiment_rescue_regression_screen.py diagnose --output data/r --year 5 --discovery $discovery
    Run-Checked $python experiment_rescue_regression_screen.py screen --output data/r --year 5 --engines catboost --budget 360 --discovery $discovery
    Run-Checked $python experiment_rescue_regression_screen.py full-refit --output data/r --budget 360
    Write-Host 'Ready: C:\ar-reg\data\r\full_refit\submission.csv'
}
```

Запуск заново требует нового output, чтобы не затереть доказательства. Ошибка
одного шага останавливает последовательность. Для проверки отдельного уже
подготовленного full-refit используйте новый независимый output/план; не
удаляйте старые результаты ради перезапуска.

## Исходный native main с активной историей

Этот отдельный прогон нужен для сравнения с измеренным **44.569657 RMSE**:
main получает исходные 53308 строк TRAIN и сам выделяет известный Year 5
(35000 fit / 18308 outer). Команда `native --years 5` выше использует уже
урезанную задачу и создаёт дополнительный уровень разбиения; её результат
47.591119 относится к другому маршруту.

После завершения очереди выше сохраните следующий Python-блок как
`data/native_original_reproduce.py`. Он одинаков для Linux и Windows;
запускайте из корня `ar-reg` тем же интерпретатором. Каталог `data/r/native_full`
должен отсутствовать. Копируются только исходный task и четыре CSV, без старых
запусков. Решатель получает 900 секунд; внешние 30 секунд нужны для завершения
и очистки. `run_python` задаёт восемь CPU-потоков и запрашивает наблюдаемый
лимит 32 GiB; недоступная телеметрия не доказывает соблюдение памяти.

```python
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
from agent.execution import run_python
from agent.task_spec import load_task
from experiment_rescue_regression_screen import CSV, digest, plan, source, write

r = Path('data/r').resolve()
base = Path('../ar-base').resolve()
before = source(base)
if before['dirty'] or before['head'] != 'd05572846369f3b1e9cfa33f4008dda028db4fe0':
    raise ValueError('Expected the clean pinned incumbent checkout')
frozen = plan(r)
task = Path(frozen['source_task'])
spec = load_task(task)
files = [task, *(spec.data_dir / name for name in
    ('train.csv', 'test.csv', 'data_dictionary.csv', 'sample_submission.csv'))]
for path in files:
    if digest(path) != frozen['source_files'][path.name]:
        raise ValueError('Original input changed: ' + str(path))
work = r / 'native_full'
work.mkdir(exist_ok=False)
for path in files:
    destination = work / path.relative_to(task.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
args = [str(base / 'solution.py'), str(work / 'task.md'),
        '--mode', 'offline', '--time-budget', '900']
code = ('import runpy,sys\n' + f'sys.path.insert(0,{str(base)!r})\n'
        + f'sys.argv={args!r}\nrunpy.run_path(sys.argv[0],run_name="__main__")\n')
execution = run_python(code, work, work / 'launcher', 0,
    timeout=930, max_memory_bytes=32 * 1024**3)
if source(base) != before:
    raise ValueError('Incumbent source changed during execution')
write(work / 'launcher_summary.json', {'source': before, 'execution': execution})
if (execution.get('exit_code') != 0 or execution.get('timed_out')
        or execution.get('memory_limit_exceeded') or execution.get('error')):
    raise RuntimeError('Inspect native_full/launcher_summary.json')
test = pd.read_csv(spec.data_dir / 'test.csv', **CSV)
prediction = pd.read_csv(work / 'submission.csv', **CSV)
spec.validate_frame(prediction, test[spec.id_column].tolist())
if not np.isfinite(pd.to_numeric(prediction[spec.target_column])).all():
    raise ValueError('Non-finite final prediction')
write(work / 'execution_summary.json', {'source': before, 'budget_seconds': 900,
    'execution': execution, 'api_mode': 'offline', 'api_calls': 0,
    'fit_rows': len(pd.read_csv(spec.data_dir / 'train.csv', usecols=[spec.id_column])),
    'predict_rows': len(test), 'ids_exact_order': True, 'finite': True,
    'prediction_sha256': digest(work / 'submission.csv'), 'quality_metric': None})
for report in work.glob('.agent_runs/*/run.json'):
    print('Native report:', report)
print('Full TEST prediction:', work / 'submission.csv')
```

Linux:

```bash
../ar-env/bin/python data/native_original_reproduce.py
```

PowerShell:

```powershell
& 'C:\ar-env\Scripts\python.exe' data\native_original_reproduce.py
if ($LASTEXITCODE -ne 0) { throw "Native original-task run failed: $LASTEXITCODE" }
```

Сравнивайте outer-предсказания из нового `run.json`, где история действительно
исполнялась, а не `native_full/submission.csv`: последний содержит 22449 строк
конкурсного TEST без доступных ответов. Значение 44.569657 — результат сохранённого
запуска, не обещание побитного совпадения адаптивного поиска на другом CPU.
Outer использовался для принятия истории и не является независимым тестом.

## Native финальная проверка

Новый кандидат ещё не подключён к общему host. Поэтому команда
`solution.py --new-candidate ...` здесь намеренно не выдумывается. После получения
точного integration SHA/переключателя от капитана выполнить две arms на одном
этом SHA, с одинаковым лимитом 900 секунд, отдельно от исходного-main system A/B.
Наличие нового исходного файла в ZIP само по себе не означает, что он исполнялся.
