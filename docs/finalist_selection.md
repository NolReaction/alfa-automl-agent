# Experimental finalist selection v1

`--finalist-selection` keeps a measured runner-up available for shared INNER
rechecks before the research dialogue closes. It is opt-in: production defaults
are unchanged. It does not add API calls, create new recipes during rechecks,
increase the original deadline, or establish a quality improvement.

## Which recipes participate

The host retains at most six frozen, successfully measured recipes. A shortlist
contains at most three distinct code/recipe identities **in total**:

- the original baseline;
- the current incumbent candidate, when distinct from the baseline;
- the strongest remaining measured alternative.

Only finite canonical INNER metrics computed by the host are admitted, after
stage-cost and prospective finalization checks. Candidate-written scores are
not evidence. Repeated observations cannot overwrite a frozen recipe's canonical
score. Family diversity breaks equal-score ties only; it cannot promote a
worse score. Pool eviction cannot remove the caller-owned baseline/incumbent.

The strongest alternative can become unaffordable as time passes. It is then
dropped; v1 does **not** search for a cheaper, lower-ranked substitute. A shortlist
can therefore contain fewer than three recipes. Identical identities, unavailable
task semantics and insufficient admissible splits can prevent shared rechecks.

## Shared rechecks and selection

`--stability-splits 2` means the canonical split plus one additional INNER split;
`3` requests two additional splits. Every shortlisted recipe receives the same
additional fit/held rows. Existing grouped/time boundaries remain authoritative.
The shortlist and canonical metrics are frozen before these extra measurements.

Only complete shared rounds enter the paired mean comparison. A failed or
interrupted partial round cannot reward the only successful recipe. Earlier
complete rounds remain usable; the report distinguishes skipped, partial and
completed checks. At least one complete extra round is required to change the
incumbent. Equal means preserve the existing choice. The selected recipe retains
its original canonical metric; a repeat mean is not substituted into that field.

Neither shortlist construction nor these rechecks use OUTER answers. Afterwards,
the existing frozen baseline/candidate OUTER admission and full-refit protocol
still applies; this feature does not replace those safeguards or create an
independent estimate of the finally chosen system's quality.

## Deadline and finalization reserve

All work uses the original task deadline, at most 10,500 seconds. Each candidate
worker remains capped at 1,800 seconds. Every prospective winner retains its own
OUTER/full-refit plan; available research time uses the most restrictive of these
plans. An inexpensive incumbent cannot lend its timing estimate to an expensive
alternative. The baseline recovery plan is retained without double-counting it.

Before another research trial, the coordinator also protects the estimated cost
of requested shared repeats. These estimates use measured recipe timings and
current split sizes. They are heuristics, not guarantees that every repeat or
refit will finish. Reserve checks refresh during rechecks; a newly infeasible
winner is not adopted. No failure starts a fresh training budget.

Inspect `run.json` for `finalist_budget`, `inner_stability`, frozen hashes,
canonical metrics, repeat scores/timeouts, reasons, selection changes and the
normal publication/receipt evidence. `finalization_reserve`, `repeat_failed` and
`fewer_admissible_splits` are outcomes to retain, not reasons to hide a run.
Source/input/frozen-identity failures remain integrity failures outside optional
fit-error handling; they must not become successful checkpoint recovery.

## Run locally

The flag requires `--stability-splits 2` or `3`. Use v3 auto validation and omit
manual `--validation random/grouped/time` or `--time-column` overrides. A task
must have supported v3 semantics. Offline mode performs no provider requests.

```powershell
.\.venv\Scripts\python.exe solution.py --help
.\.venv\Scripts\python.exe solution.py data/regression/task.md --mode offline --time-budget 900 --finalist-selection --stability-splits 2
```

```bash
.venv/bin/python solution.py data/classification/task.md --mode offline --time-budget 900 --finalist-selection --stability-splits 3
```

These commands exercise the normal solver; they do not guarantee three eligible
finalists or enough spare time for all requested rechecks. Keep seeds, inputs,
dependencies and inclusive budgets identical when comparing enabled/disabled
versions. Do not infer lower resource usage from wall-clock time alone.

## Native functional smoke

`experiment_finalist_smoke.py` exercises real INNER measurements, shared checks,
OUTER/full refit and publication receipts on two known synthetic cases:

```powershell
.\.venv\Scripts\python.exe experiment_finalist_smoke.py --output data/experiments/finalists/smoke-001 --time-budget 180
```

The output directory must be new. Read `functional_report.json`, per-case
statuses and exact receipts; the command alone is not a passing result. The
controlled mean/majority prior is deliberately weak to expose protocol behavior.
Audit labels are not scored. This is neither an incumbent quality comparison nor
independent confirmation, and it does not justify enabling the flag by default.

The [2026-10-21 native evidence](evidence/finalist-native-20261021.json) records
both cases passing in 18.536 seconds total, with three finalists, two complete
INNER rounds and a 600-row full refit per case. Source hashes matched before and
after execution. Peak CPU/thread/memory limits were not verified in this run.

For real public datasets, Windows CPU experiments, separate GPU exploration and
frozen confirmation, use the external [benchmark laboratory assignment](team/SECOND_AI_BENCHMARKS.md).
That laboratory is separate work; its planned CLI and experiments are not
implemented or completed merely by adding this finalist-selection mode.
