# Round 4 LLM: preregistered bounded plan

Base and control: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`, verified equal
to origin/main before implementation on 2026-10-23 MSK. Branch
`work/round4-llm-v1`. Do not change this comparison base mid-series.
This plan is recorded before new candidate fits or screening in this round.

## Hypotheses and interfaces

1. A bounded snapshot of existing trusted INNER information can eliminate
   information-gathering needs: task/prediction schema, allowed features,
   baseline identity/settings, measured INNER aggregates, timing and tools.
   New profiles use supplied INNER-fit features only, with a small explicit
   row/cell/time cap and no training. Never infer provenance from a scope label.
2. A small pure next-action policy can request all three candidate files and
   stop repeated exact identities or persistent non-experiments. Identity is
   code hash plus canonical-recipe hash, never model-family similarity.
3. Stopping a demonstrated repeat loop may reduce wasted requests/executions
   while preserving the existing successful INNER result and finalization.
   Scripted replies cannot prove that extra context changes live Qwen behavior.

The helper stays inactive in production. The offline harness invokes it through
a narrow test-only adapter before the original request/time/token guards.
No client wrapper changes a payload after those guards. Original host,
baseline algorithms, regression rollout, splits and finalization remain owned
by the captain.

## Screening and stopping rules

- Offline protocol screening only: deterministic tiny synthetic tasks, existing
  host split/seed construction, identical input bytes and split hashes per pair.
  Every scenario explicitly records its seed/split; no adaptive tuning of folds.
- Old arm is the pinned main request policy; new arm uses the helper with the
  same supplied facts, scripted replies, attempt cap and wall-clock budget.
  Scripts do not reward seeing the new hint. Run arms sequentially, <=8 threads.
- Use at most four mock attempts per arm, <=90 seconds per tiny host run, and
  a bounded finite set of protocol cases. Cheap fixed baseline fixtures must be
  labeled protocol-only; they cannot stand in for full-main quality control.
- Required distinct risks: useful candidate; analysis without artifacts;
  executable without training; exact duplicate; missing usage; truncation;
  invalid labels/probabilities; 504 with exhausted retry time; invalid scope;
  INNER-valid outer loser; local winner after a Qwen trial.
- Primary protocol criterion: fewer redundant calls/executions in a repeat
  loop, no false verified-fit claims, preserved CSV/finalization and guards.
  Report equal/negative results as such. If stopping suppresses a useful late
  candidate, record the counterexample; do not keep retuning on that fixture.
- At most one candidate policy and one documented corrective iteration for a
  concrete invariant failure; no expanding search for favorable examples.
- Before screening, include one explicit stop-policy counterexample: two
  analysis-only replies followed by a valid candidate and completion. The
  proposed two-nontrial stopping rule can suppress this late success. Record
  that loss if observed, without tuning another threshold on the same fixture.

## Measurements and live boundary

Measure payload bytes, reservations vs provider usage, attempts/responses,
valid artifacts, actual candidate invocations, verified INNER results, distinct
identities, duplicate counts, time to first verified INNER and total wall time.
An invocation or exit=0 is not proof of training. Internal physical fits are
unknown unless known fixture instrumentation measures them; record null with
reason otherwise. Save source/input/split/recipe hashes and environment versions.

Check standard local API configuration without printing values. If missing,
set `live_status=not_executed_no_credentials`, finish all offline work and PR.
If present, run only the existing bounded live integration gate with its one
shared ceiling (6 attempts, 30k estimated tokens, 600 seconds). No second paid
series or separate per-arm allowance. Reserve the 8192-token v3 response before
admission. Live paired utility remains unmeasured if two useful arms do not fit.

Quality comparison is a separate captain-owned experiment: current main's
unaltered offline route vs the same full agent with the proposed Qwen policy,
identical hidden-answer task and total wall-clock/CPU budget, including baseline,
network wait, experiments, outer and refit. Do not substitute fixture metrics or
development outer for that unavailable quality comparison.
