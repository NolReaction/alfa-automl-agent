# Regression rescue: predeclared development plan

Base: `d05572846369f3b1e9cfa33f4008dda028db4fe0`. Branch: `work/rescue-regression-v2`.
Declared before new model fitting on 2026-10-22. No API calls; CPU fits execute serially.

## Evaluation and separation

The primary diagnostic reproduction is the current native main, 900 seconds,
fit years 1–3 and predict year 4. Additional development folds predict years 2
and 3 from all strictly earlier years. All three folds are fixed before tuning.
Years 3 and 4 have also appeared in repository research: these are rolling TRAIN
development slices, not blind external tests. Year 5 is an already exposed
compatibility check, run only after the recipe/algorithm is frozen. TEST labels
are unavailable and are never sought. Input TRAIN SHA256:
`096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`.

| Predict period | Fit rows | Predict rows | Role |
| --- | ---: | ---: | --- |
| 2 | 2117 | 9058 | Single-period fallback development |
| 3 | 11175 | 10879 | Rolling development |
| 4 | 22054 | 12946 | Primary diagnostic development |
| 5 | 35000 | 18308 | Known compatibility after freeze |

Native incumbent uses an immutable separate checkout. Every invocation has a
fresh output directory and preserved artifacts. Isolated candidates use the
existing frozen-candidate host with original-scale RMSE recalculated from saved
CSV and host-only answers. Native incumbent versus isolated candidate is a
screening comparison, not an equal-scheduler causal A/B. Equal 900-second native
on/off runs require the captain's integration SHA and remain pending until then.

## Initial hypotheses (bounded to three)

1. Stronger original-scale base helps the no-history rows. Two CPU configurations:
   CatBoost depth 6, learning_rate .05, l2_leaf_reg 5; LightGBM num_leaves 31,
   max_depth -1, min_child_samples 40, learning_rate .04, reg_lambda 5,
   colsample_bytree .9. Both cap at 2000 iterations, stopping patience 100,
   seed inherited from the repository, eight threads. Last fit period selects
   the iteration count; a fresh model refits all permitted fit rows. When fit
   has one period, use the fixed 400-iteration fallback without early stopping.
   Each isolated invocation has at most 360 seconds including preparation.
2. Last/decayed history can improve over mean history. Compare base alone,
   fixed mean at .5, and fit-calibrated history. Calibration uses the same
   fit-internal temporal base predictions, never host-held labels. Predefined
   choices are no history plus mean/last/decay × weights .25/.5/.75 × count
   shrinkage 0/1 (19 choices, no additional model fits). Decay is .5 per numeric
   period gap. Ambiguous past keys under `past_unique_only`, missing keys and
   invalid periods preserve the base. No-history calibration or insufficient
   eligible rows falls back to the existing fixed mean .5 policy, disclosed.
3. Only if bounded discovery blocks the rolling folds, examine a larger but
   explicit work/check allowance. Preserve feature-only input, complete top-one
   certification, width ≤3, ≤12 screened columns, no partial-search activation,
   and an 8-second/1-GiB discovery worker. Any change is opt-in and separately
   ablated. Never encode the observed public key names in runtime.

No log target, clipping, residual model, additional family or large ensemble is
in this initial screen. Expansion requires an explicit evidence-based amendment.

## Frozen diagnostics

Report n, row share, SSE, SSE share, RMSE and MAE for each partition separately:
reference-history eligible/absent/ambiguous (same masks for all arms); past rows
0/1/2/3+; recency none/one/two-or-more numeric periods; target bands at fit p50,
p90,p99; size bands at fit p50,p90 plus missing; missing feature count 0/1–5/6+;
new/rare/common categories (rare fit count < max(20, ceil(.001*n_fit)));
period; top three fit levels plus other for at most two declared categorical
features. Candidate coverage is reported separately. Dictionary confirms
`floor_area` is area, while `Year_Factor` is an anonymized ordinal year, so
subtracting construction year from it is not a valid age feature.

## Decision and evidence

Retain simple candidates with improvement on a majority of predeclared folds;
explain any large regression. Freeze before year 5. Final go also requires
native execution/publication, paired timing and full refit. Until integration,
report `integration_pending`, never native victory. Record input/split/source/
recipe/output hashes, IDs and finite values, CPU time/observed RSS, timeout and
cleanup status. Unobserved telemetry is unknown, not zero. No main merge.
