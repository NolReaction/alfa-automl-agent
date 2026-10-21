# Opt-in history runtime experiment

This harness compares native `run_agent_v3(..., client=None, local_only=True)`
with history disabled and with one explicit history configuration at blend
weight 0.5. Both arms use the existing AutoML startup and local continuation,
the legacy search profile, and the same walltime budget. No provider requests
are made. History remains opt-in; these experiments do not change a default.

## Frozen method

Development is four synthetic cases at seed 131: repeating entity signal,
zero history coverage, colliding keys without useful history signal, and IID
regression whose native policy must skip history. Each has 240 training rows
and 60 external audit rows. Seed 379 is reserved for future development.
Confirmation is entity signal and colliding keys at seeds 2203 and 4001.
The 16 runs have 90 seconds each and share an active coordinator budget of
1800 seconds. A pair is admitted only when both full run budgets remain.

The immutable runtime is local commit
`782d738a5992ea5fb93767c7da5e5e2d7b266504`, tree
`aef1d8f387f621c7f6d33e84878081b34a1cbf80`. Its reproducible remote
equivalent is commit `7cfa8c98a11e64ee09ab343a5917f2294ccc3e77` with the
same whole Git tree. The manifest records exact source bytes, harness bytes,
package versions, task files, literal-ID splits, configurations and audit
answer hashes before fitting. Confirmation also requires a separate method
seal after the completed development report and before confirmation starts.
Source, input, package or method changes stop execution and re-audit.

## Execution

Use an otherwise idle host and one immutable solver checkout. For a protocol
smoke first, add `--smoke --total-seconds 220` to `prepare` and use a separate
output directory. This is one development pair, not confirmation evidence.

```bash
python experiment_history_runtime.py preflight --solver-root /path/to/frozen
python experiment_history_runtime.py prepare --solver-root /path/to/frozen --out-dir data/experiments/history_runtime/v1 --time-budget 90 --memory-mib 6144 --total-seconds 1800
python experiment_history_runtime.py run --out-dir data/experiments/history_runtime/v1 --stage development
python experiment_history_runtime.py audit --out-dir data/experiments/history_runtime/v1 --stage development
python experiment_history_runtime.py freeze-confirmation --out-dir data/experiments/history_runtime/v1
python experiment_history_runtime.py run --out-dir data/experiments/history_runtime/v1 --stage confirmation
python experiment_history_runtime.py audit --out-dir data/experiments/history_runtime/v1 --stage confirmation
```

`--public-task /path/to/regression/task.md` prepares a separate optional known
public TRAIN sanity pair: fit years before 5, audit year 5. Use 300 seconds per
arm and a separate plan/budget. The six historical keys live explicitly in
this experiment, not in the runtime. This public case was previously studied
and is not an independent confirmation case.

## Evidence and interpretation

Host audit answers stay outside each solver task directory. A fresh process
imports the frozen solver and reports the exact returned run directory.
Re-audit executes no candidate code. It verifies API counters, native startup
scope, split IDs, frozen baseline hashes, inner history feedback, selected
outer and full-data fit rows, prediction schemas, nonce/code/recipe/output
receipts, published CSV, and the last valid checkpoint. Persisted trial proofs
must match the newly recomputed proofs. Ordinary subprocess isolation is not
an operating-system security sandbox.

The primary paired result is external RMSE under equal total walltime.
`baseline_recipe_match` is recorded for every pair: timed search can select
different startup recipes. A mismatch does not invalidate the end-to-end
comparison, but prevents attributing the difference solely to history. Even
matching baselines can have different local candidate admissions because a
history trial consumes time. Zero coverage is a measured trial cost, not an
automatic pre-fit success or skip. An IID skip checks policy behavior.

The current local host has 8 CPU and an 8 GiB cgroup, unlike the platform's
32 GiB. A requested 6144 MiB process-tree cap is recorded. When process
inventory is unavailable, neither measured peak RSS nor cap enforcement is
claimed. These bounded developer runs are not a three-hour platform or
private-score qualification.

`tests/test_history_runtime_experiment.py` uses a fake worker with synthetic
receipts to test case determinism, host-answer separation, method freezing,
version/input/source changes, tampering, baseline mismatch handling and
whole-pair budget censorship. Its scores are fixture values and provide no
evidence of ML quality. Real results belong in a separate evidence report.
