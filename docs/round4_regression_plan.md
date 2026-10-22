# Round 4: preregistered regression development

Declared 2026-10-23 MSK before model fitting. Base and current main verified:
`e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`. Branch: `work/round4-regression-v1`.
The existing rescue-v1 is integrated; historical PR39 `integration_pending`
statements do not describe this incumbent. No changes to shared host/CLI.

## Evaluation contract

Use the existing public TRAIN (expected SHA256
`096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`).
Development outer transitions predict periods 4, then 2 and 3, from all strictly
earlier periods. All are exposed development, not blind tests. Seed42; CPU8;
one heavy invocation at a time. Each invocation has the same 180-second cap
including preparation, internal calibration, prediction and fresh full refit.
Requested memory cap8GiB; measured RSS/CPU/cleanup remain null when unavailable.

For outer period t, host INNER-fit is the supplied fit excluding its latest
period (t-1). Discover once on those features only, using current 8M/48 defaults
(`extended=False`), freeze the same configuration for every arm. Candidate
internal calibration independently uses period t-1 inside its supplied fit.
No extra derived-task native nesting; no 12M/192 schema and no rediscovery in
fit_predict. A missing schema makes history partitions not applicable; do not
guess keys from prior public results. Native main has already integrated v1;
the control here is that fixed candidate, not the entire adaptive scheduler.

Year5 is used once after final algorithm selection for compatibility, with
identical newly frozen host INNER-fit schema. After that, no retuning. Full
TRAIN refit reuses the exact selected source and known5 recipe/config; no TEST
answers. Captain owns subsequent equal-budget native900s A/B and main merge.

## Hypotheses and fixed screen (at most 11 configurations)

I: current CatBoost rescue-v1, depth6, lr.05, l2=5, max2000, patience100,
single-period fallback400, minimum refit50, calibrated history (19 fixed options).
H1 tests at most four new coordinated base configurations, same other limits:

| Arm | Depth | Learning rate | L2 | Rationale |
| --- | ---: | ---: | ---: | --- |
| B4 | 4 | .05 | 10 | Reduce variance on cold/missing objects |
| B5 | 5 | .04 | 10 | Moderate complexity with stronger regularization |
| B7 | 7 | .04 | 10 | Allow bounded nonlinear interactions |
| B6R | 6 | .03 | 20 | Slower learning and stronger regularization |

H2 tests Fmissing, Ffrequency and Fboth on the unchanged incumbent base.
Numeric missing flags preserve information lost in median imputation. Relative
category frequencies represent fit support while preserving original categories;
unseen categories map to0. At most16 categorical columns,512 levels per column;
oversized maps abstain. The adapter is fitted independently inside each internal
training fold and again on full refit. No target-derived features or TEST-based
selection. Column choices follow declared roles and bounded deterministic rules.

One C combination is permitted only if both a H1 base and a H2 block improve
period4 RMSE; combine the best of each (ties within0.2% choose fewer features,
then shallower model). Otherwise skip C. No full Cartesian grid.

H3 is diagnostic: LGBlegacy and LGBaligned use the existing fixed LightGBM
recipe. If best_iteration<50, aligned retrains only earlier calibration rows at
the actual selected50 trees before history calibration (one additional physical
fit, no budget extension). Its final full fit still uses50. No expansion of the
19-option history grid or rolling folds without evidence. History mean and
calibrated alternatives are recorded from identical fits for every arm. A null
schema can make H3 uninformative; record this rather than forcing activation.

Total: I +4 H1 +3 H2 +optional C +2 LGB diagnostics = at most11.
No log target, clipping, residual model, cold-only model or new model family.

## Promotion, stopping and evidence

Run the screen on period4. Promote at most two promising H1/H2/C configurations
to periods2/3 along with a fresh incumbent at the same cap. Require lower RMSE
on a majority of applicable development periods, aggregate SSE improvement,
and no unexplained >5% relative RMSE loss on another period or >10% cold-group
SSE increase. Inspect row counts and absolute SSE; overlapping segment shares
are not additive. Differences below0.2% are treated as inconclusive and favor
the simpler model. Repeat seeds only if a material stochastic/timing sensitivity
is demonstrated, not to search a favorable seed. Two distinct failures stop a
hypothesis. If no candidate meets the gate, retain incumbent and deliver negative
evidence; do not consume Year5 to tune an unqualified model.

Every summary records input/source/split/recipe/output SHA256, environment,
actual started/completed physical fits, fit rows and durations, cap and resource
availability. Failed or unavailable values are null with reasons. Save CSVs and
four disjoint history groups where a schema exists: eligible, ambiguous past,
valid key without past, incomplete key/invalid period. Other partitions: unseen
categories, numeric missing counts, target bands with fit-only quantiles, and
largest fit-defined categorical segments. Record tree counts, fallback reason,
history selection, calibration support, internal-fit and fresh-refit time.

Preserve rescue-v1 bytes and semantics. New behavior has explicit round4 family
and frozen feature/alignment settings. Tests cover fold locality, strict past,
predict row/batch invariance, no mutation, normalized recipe, budget/fallback,
fresh refit and extracted flat-package imports. Native admission remains a
captain integration step; published evidence distinguishes that from isolated
algorithm readiness. No API, platform submission, runtime downloads or models
in Git. A new discovery budget or host hook requires an integration request.
