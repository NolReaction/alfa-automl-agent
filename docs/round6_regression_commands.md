# Round6 reproduction and evidence commands

Measured source: local `d6354a4d6cf446ac895b1ca298527ff8fea7666b`, published
`0b721203077c12777e72d4e294b47f7fa9f3dbcf`, identical tree
`ce3ed45f7ecc85b3017068e7780e849612ed49cd`. Use the published implementation
commit for reproduction. Later report commits do not change measured code.
Existing input files must match `docs/round6_regression_plan.json` hashes.
A reproduction is a separate development run; it does not replace this evidence.
Do not reuse existing output directories or run a second queue concurrently.

## Measured Linux environment and sequence

Python executable used:
`/workspace/scratch/b69ed5d5758a/round5-env/bin/python` (3.12.14).
Packages: numpy1.26.4, scipy1.12.0, pandas2.2.3, scikit-learn1.5.2,
CatBoost1.2.8, LightGBM4.6.0 (dependency only), psutil6.1.1.
The ZIP contains the full `pip freeze`. No dependencies were changed.

```bash
# Resolve an existing pinned environment before entering a new worktree.
PY="$(realpath .venv/bin/python)"
git fetch origin work/round6-regression-calibration-v1
git worktree add --detach ../alfa-round6-repro 0b721203077c12777e72d4e294b47f7fa9f3dbcf
cd ../alfa-round6-repro
# Place the already authorized, hash-matching task and participants data
# under data/regression; do not download or substitute a different dataset.
OUT=data/round6-repro
SMOKE="$(realpath ..)/round6-flat-smoke-repro"
H=experiment_round6_regression_calibration.py

"$PY" -m unittest discover -s tests -p 'test_round6_regression_*.py' -v
"$PY" "$H" prepare --task data/regression/task.md \
  --period-column Year_Factor --size-column floor_area \
  --published-commit 0b721203077c12777e72d4e294b47f7fa9f3dbcf --output "$OUT"
# Register the measured model-free receipt described below before smoke.
"$PY" "$H" register-checks --receipt /absolute/path/checks-receipt.json --output "$OUT"

# Execute these three commands back-to-back under the same 195-second reservation.
"$PY" "$H" reserve-validation --output "$OUT"
ROUND6_NATIVE_SMOKE_OUTPUT="$SMOKE" "$PY" -m unittest discover -s tests \
  -p 'test_round6_regression_native.py' -v
# Always register, including if the smoke process returned nonzero.
"$PY" "$H" register-validation --smoke-directory "$SMOKE" --output "$OUT"

"$PY" "$H" queue --output "$OUT"
"$PY" "$H" audit --output "$OUT"
"$PY" "$H" export --output "$OUT" --destination ../round6-repro-original-export.zip
"$PY" "$H" compact --output "$OUT" --destination ../round6-repro-compact
```

The actual root process called `reserve_validation`, `run_flat_smoke` and
`register_validation` in one Python process using `try/finally`, ensuring
registration even on failure. The CLI commands above have the same behavior
when executed without pauses; both receipt elapsed and the original reservation
deadline must pass. Native smoke is a portability/contract test, not admission.

`queue` performs discovery4 once, screen I/S20/S50/A20 and fixed promotion.
Only if a screen finalist exists does it run fresh cap360 I plus at most two
finalists on periods2/3/4, stopping a finalist after two material failures.
This measured run promoted nobody, so no confirmation, freeze/full fit,
original TEST or Year5 command was executed. There is no full-fit command in
this Round6 harness. No runtime/default family or platform is changed.

## Model-free receipt

Measure the entire model-free subprocess wall time, including any auxiliary
recipe/hash checks, and capture `validation_source_hashes()` immediately before
and after. Preserve stdout/stderr. Require equality of the two hash maps.
The exact JSON schema accepted by `register-checks` is:

```json
{
  "version": 1,
  "kind": "model_free_checks",
  "label": "preregistered_contract_tests",
  "elapsed_seconds": 0.0,
  "exit_code": 0,
  "command": "actual measured command",
  "source_hashes": {}
}
```

The zero/empty entries are schema placeholders, not valid measurement evidence.
Replace them with observed elapsed time, subprocess status, exact command and
`experiment_round6_regression_calibration.validation_source_hashes()` output.
The final measured batch's real receipt is included in the ZIP. Native smoke
must stay disabled during model-free discovery; enable its output environment
variable only after `reserve-validation`. Never substitute an estimated fit
count for observed start/completion events.

## Windows PowerShell

Resolve the intended interpreter before moving between worktrees. If it is not
available, create a separate Python3.12 environment with
`C:\Windows\py.exe -3.12 -m venv <new-path>` and install the frozen requirements
there; record setup costs before a new measured run. Do not modify the existing
night-run environment or its checkout at `8a1c243`.

```powershell
$Python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $Python --version
git fetch origin work/round6-regression-calibration-v1
git worktree add --detach ..\alfa-round6-repro 0b721203077c12777e72d4e294b47f7fa9f3dbcf
Set-Location ..\alfa-round6-repro
$Out = 'data/round6-repro'
$Smoke = Join-Path (Split-Path (Get-Location).Path -Parent) 'round6-flat-smoke-repro'
$Harness = 'experiment_round6_regression_calibration.py'

& $Python -m unittest discover -s tests -p 'test_round6_regression_*.py' -v
& $Python $Harness prepare --task data/regression/task.md `
  --period-column Year_Factor --size-column floor_area `
  --published-commit 0b721203077c12777e72d4e294b47f7fa9f3dbcf --output $Out
& $Python $Harness register-checks --receipt 'C:\absolute\checks-receipt.json' --output $Out

& $Python $Harness reserve-validation --output $Out
try {
  $env:ROUND6_NATIVE_SMOKE_OUTPUT = $Smoke
  & $Python -m unittest discover -s tests -p 'test_round6_regression_native.py' -v
} finally {
  Remove-Item Env:\ROUND6_NATIVE_SMOKE_OUTPUT -ErrorAction SilentlyContinue
  & $Python $Harness register-validation --smoke-directory $Smoke --output $Out
}
& $Python $Harness queue --output $Out
& $Python $Harness audit --output $Out
& $Python $Harness export --output $Out --destination '..\round6-repro-original-export.zip'
& $Python $Harness compact --output $Out --destination '..\round6-repro-compact'
```

Check each command's exit status before proceeding; retain failed evidence.
At most one documented technical model retry may use the same source, seed,
recipe and cap in a new folder. This study used none. An audit-only correction
must not repeat completed models.

## Independent audit and export supplement

The bundle includes `audit/independent_v1.py` and `independent_v2.py`, the original
failed and final successful reports and both cost receipts. V2 uses no project
imports. Run it against the untouched measured checkout and output directory:

```bash
"$PY" independent_v2.py --experiment /absolute/round6-experiment \
  --checkout /absolute/measured-checkout --output /absolute/fresh-audit.json
```

Charge its measured subprocess cost with `kind=audit_only_checks`. Retain failed
reports. Do not copy the data-free ZIP over the experiment input tree: it
intentionally omits raw inputs/answers. Independent rescoring needs the original
authorized host-only answers matching the recorded hashes.

Preserve the original measured export and its manifest. For the final evidence
bundle, add the actual immutable `frozen/research_candidate.py`,
`frozen/research_recipe.json` and `frozen/frozen_candidate.json` per run; the
measured exporter names differ. Add audit reports/scripts, publication record,
full environment and test receipts, then regenerate an explicit member
SHA256/size manifest and verify CRC and every member hash. The final bundle
preserves the original measured ZIP/manifest/receipt as provenance. Never add
raw TRAIN/TEST, answers, fit/predict/context inputs, models or raw A/B groups.
The final archive's own hash and packaging-cost receipt live in the final Git
handoff to avoid a circular archive hash.
