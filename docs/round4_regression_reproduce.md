# Reproduce the round-4 negative result

This is a fixed replay of the completed experiment, when independently needed.
It is not an invitation to tune again on known Year5. The frequency-only
finalist was rejected: compatibility RMSE 47.640123 versus incumbent 39.471458.
Keep the integrated rescue-v1 incumbent.

Base: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`.
All measured code: `be060588e820eec54c2cbacc96f998d4bcbb75cf`.
Later documentation commits do not change that measured algorithm. Use a clean
checkout and a new output directory. Run one CPU queue only, with eight threads;
do not run these fits concurrently with another training job.

## Environment and data

Linux, from an unused directory:

```bash
set -e
git clone https://github.com/NolReaction/alfa-automl-agent.git ar4
cd ar4
git switch --detach be060588e820eec54c2cbacc96f998d4bcbb75cf
python3.12 -m venv ../ar4-env
../ar4-env/bin/python -c "from pathlib import Path; import re; Path('data').mkdir(exist_ok=True); Path('data/round4-platform-constraints.txt').write_text(re.sub(r'\[[^\]]+\]', '', Path('requirements.platform.txt').read_text(encoding='utf-8')), encoding='utf-8')"
../ar4-env/bin/python -m pip install -r requirements.txt -c data/round4-platform-constraints.txt
../ar4-env/bin/python prepare_data.py regression
../ar4-env/bin/python -m unittest discover -s tests -p 'test_round4_regression*.py' -q
../ar4-env/bin/python -m unittest discover -s tests -p 'test_rescue_regression*.py' -q
```

PowerShell, using unused short directories:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    function Run-Checked {
        $program = $args[0]
        $commandArgs = @($args | Select-Object -Skip 1)
        & $program @commandArgs
        if ($LASTEXITCODE -ne 0) { throw "$program failed: $LASTEXITCODE" }
    }
    foreach ($path in @('C:\ar4', 'C:\ar4-env')) {
        if (Test-Path $path) { throw "Choose unused directories: $path exists" }
    }
    Run-Checked git clone https://github.com/NolReaction/alfa-automl-agent.git C:\ar4
    Set-Location C:\ar4
    Run-Checked git switch --detach be060588e820eec54c2cbacc96f998d4bcbb75cf
    Run-Checked 'C:\Windows\py.exe' -3.12 -m venv C:\ar4-env
    $python = 'C:\ar4-env\Scripts\python.exe'
    Run-Checked $python -c "from pathlib import Path; import re; Path('data').mkdir(exist_ok=True); Path('data/round4-platform-constraints.txt').write_text(re.sub(r'\[[^\]]+\]', '', Path('requirements.platform.txt').read_text(encoding='utf-8')), encoding='utf-8')"
    Run-Checked $python -m pip install -r requirements.txt -c data/round4-platform-constraints.txt
    Run-Checked $python prepare_data.py regression
    Run-Checked $python -m unittest discover -s tests -p 'test_round4_regression*.py' -q
    Run-Checked $python -m unittest discover -s tests -p 'test_rescue_regression*.py' -q
}
```

Data downloading here is developer preparation of the provided public task;
the candidate performs no downloads or API calls. Expected original TRAIN SHA256:
`096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`.
The frozen plan records all input and split hashes. The installed environment
is the documented runtime subset, not the complete contest image.
The official list contains extras, which pip rejects in constraints files.
The commands therefore strip bracketed extras only in an ignored temporary
constraints copy, matching the measured setup. Repository dependency files
remain unchanged; this is not a runtime package installation.

## Fixed sequential replay

Save the following block as `data/replay_round4.py` in the clean checkout.
`data/` is ignored by Git, so saving the driver does not invalidate source
identity. Launch it from the repository root. It uses the existing measured
harness functions and exactly the recorded recipes, with no search expansion.
All invocations have the same 180-second cap and requested sampled 8GiB cap.
Unknown resource measurements remain unknown.

```python
from copy import deepcopy
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path.cwd()))
from agent.regression_candidate import candidate_source as incumbent_source
from agent.regression_candidate import make_regression_recipe
from agent.round4_regression_candidate import candidate_source, make_round4_recipe
from experiment_round4_regression import prepare, discover, run, freeze, full_refit, schema, write

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=('prepare', 'screen', 'confirm', 'freeze', 'known5', 'full-refit'))
args = parser.parse_args()
output = Path('data/round4-replay')
recipes = output / 'recipes'
order = ('B4', 'B5', 'B7', 'B6R', 'Fmissing', 'Ffrequency', 'Fboth', 'LGBlegacy', 'LGBaligned')

def checked(record):
    if record.get('status') != 'completed':
        raise RuntimeError('Invocation failed; inspect its preserved summary.json')
    return record

def candidate_recipe(year, **options):
    return make_round4_recipe('Year_Factor', history_config=schema(output, year)['selected_config'], **options)

if args.stage == 'prepare':
    prepare('data/regression/task.md', output, 'Year_Factor', 'floor_area')
    discover(output, 4)
    recipes.mkdir()
    (recipes / 'round4_source.py').write_text(candidate_source(), encoding='utf-8')
    (recipes / 'legacy_source.py').write_text(incumbent_source(), encoding='utf-8')
    variants = {}
    for name, depth, rate, regularization in (
            ('B4', 4, .05, 10.), ('B5', 5, .04, 10.),
            ('B7', 7, .04, 10.), ('B6R', 6, .03, 20.)):
        variants[name] = candidate_recipe(4, parameters={
            'depth': depth, 'learning_rate': rate, 'l2_leaf_reg': regularization})
    for name, missing, frequency in (
            ('Fmissing', True, False), ('Ffrequency', False, True), ('Fboth', True, True)):
        variants[name] = candidate_recipe(4, features={
            'missing_indicators': missing, 'category_frequencies': frequency})
    variants['LGBlegacy'] = make_regression_recipe('Year_Factor', engine='lightgbm',
        history_config=schema(output, 4)['selected_config'])
    variants['LGBaligned'] = candidate_recipe(4, engine='lightgbm', calibration_alignment='refit_count')
    for name in order:
        write(recipes / (name + '.json'), variants[name])
    write(output / 'screen_manifest.json', {'order': list(order), 'variants': variants, 'budget': 180, 'year': 4})
elif args.stage == 'screen':
    checked(run(output, 4, 'incumbent', 180))
    for name in order:
        source = recipes / ('legacy_source.py' if name == 'LGBlegacy' else 'round4_source.py')
        checked(run(output, 4, name, 180, source, recipes / (name + '.json')))
elif args.stage == 'confirm':
    # Replay the recorded sole finalist; do not choose new variants from replay results.
    for year in (2, 3):
        discover(output, year)
        recipe = candidate_recipe(year, features={'category_frequencies': True})
        path = recipes / ('Ffrequency_y' + str(year) + '.json')
        write(path, recipe)
        checked(run(output, year, 'incumbent', 180))
        checked(run(output, year, 'Ffrequency', 180, recipes / 'round4_source.py', path))
elif args.stage == 'freeze':
    freeze(output, 'Ffrequency', ['Ffrequency'])
elif args.stage == 'known5':
    # One frozen compatibility pair only. Do not change weights after its result.
    discover(output, 5)
    checked(run(output, 5, 'incumbent', 180))
    checked(run(output, 5, 'Ffrequency', 180))
else:
    # Executability evidence only: successful refit cannot promote the rejected model.
    checked(full_refit(output, 180))
```

The actual screen was incumbent plus nine named variants. The optional
combination was not run because no new H1 base passed the initial screen.
Default 8M/48 history discovery is frozen once per external fold; it is not
the earlier extended-discovery experiment. On development periods2/3/4 the
schema was unavailable, so history-specific results there are not informative.

Linux commands, in this order:

```bash
set -e
../ar4-env/bin/python data/replay_round4.py prepare
../ar4-env/bin/python data/replay_round4.py screen
../ar4-env/bin/python data/replay_round4.py confirm
../ar4-env/bin/python data/replay_round4.py freeze
../ar4-env/bin/python data/replay_round4.py known5
../ar4-env/bin/python data/replay_round4.py full-refit
```

PowerShell commands, from `C:\ar4`:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    foreach ($stage in @('prepare', 'screen', 'confirm', 'freeze', 'known5', 'full-refit')) {
        & 'C:\ar4-env\Scripts\python.exe' data\replay_round4.py $stage
        if ($LASTEXITCODE -ne 0) { throw "Replay failed at $stage : $LASTEXITCODE" }
    }
}
```

Never delete a failed output to hide or replace its evidence. For a deliberate
independent replay use a fresh `output` name in the driver. A failed stage
stops execution. The original final refit had one failed attempt preserved in
the evidence; the separately recorded clean-source retry completed. That
failure/retry is not a quality-selection loop and is not automatically repeated
by the commands above.

The measured Linux harness invocations, recipe generation and serial order
were executed during this round. This consolidated driver is a syntax-checked
assembly of that sequence; the complete sequence was not trained a second
time merely to test the document. PowerShell syntax is supplied for Windows;
these model experiments were measured in Linux, not remeasured on Windows.

## Recorded contract checks and recommendation

31 new focused tests and 78 incumbent regression tests passed. The separate
`flat_smoke_probe.py` in the evidence archive exercised the flattened package
outside the repository with three synthetic invocations/six physical fits.
Its 50-tree toy cap is not a full-recipe quality experiment.

The frozen JSON's historical `winner` / `native_admission_pending` labels mean
"selected after development, before compatibility". The final decision after
known5 is **rejected; retain incumbent**, as recorded in the results and
integration note. No native integration flag or host hook is being requested.
