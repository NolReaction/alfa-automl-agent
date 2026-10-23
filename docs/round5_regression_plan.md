# Round 5: preregistered fixed regression ensemble experiment

Declared 2026-10-23 MSK before any new model fit, including toy probes.
Base/main verified `66e9481cb61cb83cfecab53be6984d7d2bf829b3`; branch
`work/round5-regression-ensemble-v1`. Main will not be merged or reset.
Full parameters, input hashes and unchanged dependency hashes are frozen in
`round5_regression_plan.json`, committed together with this document.

## Hypotheses and exact fixed screen

H1 tests complementarity of existing CatBoost/LightGBM predictions; H2 tests
a global declared-numeric-only CatBoost companion. No frequency features,
category-specific routing, new objective, log target, weight tuning, seed
search or expanded history grid. Periods are anonymous ordered values; no
calendar-age interpretation. Dataset-specific names below belong only to this
developer manifest, never runtime routing.

| Order / arm | Fixed base prediction before unchanged history |
| --- | --- |
| I | Unmodified current CatBoost rescue-v1 control |
| LGB | Existing fixed LightGBM base |
| E25 | .75 CatBoost + .25 LightGBM |
| E50 | .50 CatBoost + .50 LightGBM |
| E75 | .25 CatBoost + .75 LightGBM |
| N | CatBoost using declared numeric columns only |
| N25 | .75 all-feature CatBoost + .25 numeric-only CatBoost |
| N50 | .50 all-feature CatBoost + .50 numeric-only CatBoost |

Exactly eight unique screen configurations including incumbent. CatBoost:
depth6, learning_rate=.05, l2_leaf_reg=5. LightGBM: num_leaves31, max_depth=-1,
min_child_samples40, learning_rate=.04, reg_lambda5, colsample_bytree=.9.
Both: max2000 iterations, patience100, fallback400, seed42, CPU8, original
RMSE. Existing normalized recipes/estimator construction provide the remaining
fixed defaults; source/dependency hashes bind them. Numeric-only changes only
the feature roles/frame, never model parameters or history inputs.

**Alignment choice: legacy for every component.** Internal predictions use
the selected best iteration; fresh full refit uses max(50,best), exactly as
rescue-v1. Best/final-tree mismatch is exported. No `refit_count` recheck or
hidden alignment change is introduced, so LightGBM comparison is not confounded
with the round4 diagnostic. This known limitation remains explicit when
history is later assessed. Single-period supplied fit uses fixed400 and no
internal calibration, preserving the existing history fallback.

New family `round5_regression_ensemble_v1` freezes component recipes, constant
weights/subsets, alignment, history, quotas and fail-whole-arm behavior. Its
stages reuse existing base internals: all component internal fits first;
validate identical original supplied-row fit/held sets and reindex calibration
predictions explicitly; form frozen mixture and calibrate unchanged history;
then fresh full-fit each component and form final base mixture/history output.
No fitted model crosses phases. I runs the unchanged rescue implementation.
Numeric-only receives copied frames and derived roles; HistorySignals always
receives the permitted original raw features. No input/context mutation.

## Nesting, data and resources

First screen: external period4, strictly earlier supplied fit, **180 seconds
per invocation**. Confirmation: fresh incumbent plus at most two finalists on
external periods2,3,4, each **360 seconds**. Screen180 never substitutes for
confirmation360. Those exposed development slices are not blind evaluation.
Never evaluate external known Year5, inspect its old residuals for tuning, or
create Year5 validation artifacts. TEST labels are unavailable and not sought.

For each transition, history discovery runs once on the same host INNER-fit:
supplied fit excluding its latest period. Use existing 8M cell visits /48
configs, `extended=False`, no labels/ID/held features. Freeze for every arm.
Candidate internal calibration separately uses the latest period inside its
supplied fit. Schema absence means exact history/cold groups **not assessable**,
never all-cold or passed; novel-category proxies do not replace this gate.

One original deadline covers staging, preprocessing, all internal fits,
history selection, fresh component refits and prediction. Reserve
min(8seconds,5% remaining) for history/output; allocate remaining model budget
equally to components (100% or50/50), fixed at invocation start. Component
elapsed time is cumulative across its stages; unused quota does not transfer.
Every stage is additionally capped by the original absolute deadline. Any
unfinished component fails the whole arm, with no partial-mixture fallback.
Existing single-period fallback is explicitly frozen, not failure recovery.

One sequential CPU queue, at most8 threads, no GPU/API/rented capacity.
Maximum10,800 seconds of heavy work includes model probes, failures, discovery,
tests and one permitted technical retry. Runtime process telemetry may be
unavailable; report unknown, not the memory limit as measured RSS. Planned
sampled child memory cap8GiB is not official32GB qualification. Environment:
Python3.12.14 with requirements.txt runtime subset; core official pins
numpy1.26.4/scipy1.12.0/pandas2.2.3/sklearn1.5.2/CB1.2.8/LGB4.6.0 verified by
import before fitting. Full installed package versions will be exported.

TRAIN SHA256 `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`,
53,308 rows; unlabelled TEST22,449. Other input hashes are in the paired JSON.
Preparation will freeze exact per-transition split/input identities before fit.

## Promotion, stopping and final decision

Run the eight-arm screen as one declared batch. Promote at most two arms with
>=0.2% lower period4 RMSE than the fresh I; rank by RMSE. A difference<0.2%
between candidates favors fewer component fits, then the fixed arm order.
No replacement ninth configuration. Technical failures get no score. At most
one global technical retry is allowed only for a recorded technical cause,
identical recipe/seed/cap in a new directory, preserving the first attempt.

Confirmation order is period2,3,4 with fresh equal-cap I per period. Require
RMSE improvement on a majority of applicable transitions, summed SSE lower,
and no individual relative RMSE loss>5%. Differences<0.2% are practical ties
in favor of simpler/cheaper incumbent. A finalist with two material failures
on different confirmation transitions stops; incomplete remaining gates stay
unknown and it cannot win. No weight/recipe changes or extra budget after loss.

A significant segment has >=max(100,1% held rows) and control SSE>=1% of fold
SSE. Exact history-group SSE may not worsen>10% when assessable. For novelty
or large fit-defined categories, >10% segment SSE deterioration with absolute
increase>1% of control fold SSE is a material unexplained failure and blocks
recommendation. Report row counts/absolute SSE for small groups too; overlapping
partitions are not additive. No causal story will be invented to waive a loss.
History/cold unknown may support only a **base-only frozen-None** conditional
recommendation if every other gate passes, never history/native admission.

Freeze a winner only after confirmation. Reuse the exact development schema
in full TRAIN refit; if all development schemas are None, freeze None. If they
differ, do not silently select an untested configuration: retain incumbent and
send the choice to the captain. One fresh full-TRAIN refit cap360 then predicts
original TEST with the same frozen algorithm; fit-only last-period stopping
may change tree count, but not weights/schema/algorithm. No external Year5
evaluation/discovery on full TRAIN. If no winner passes, no finalist full refit;
recommend retain-incumbent and preserve negative results.

## Evidence and checks

Each fit start is persisted before training, including engine/component/phase,
rows, completion, failure and wall duration. Record staging/preprocessing,
internal-selection, history calibration, fresh refit and prediction time;
where a native combined phase cannot be separated, state that and use null
instead of invented measurements. Capture original deadline/cap, all source,
dependency/input/split/recipe/output/receipt hashes, and all rejected attempts.
Base/mean/calibrated-history outputs share the same fits. Diagnostics: RMSE,
MAE, SSE, fit-defined target bands, novelty, missing counts, large categories;
exact four history groups only with an actual frozen schema.

Risk-focused unit tests cover strict past, fit-only preprocessing, calibration
alignment, immutability, fixed recipes and shared deadlines/failure semantics.
Exactly two planned real small E50 flat-package smoke invocations check fresh
fits and reordered/extended prediction batches outside the source checkout,
with no agent directory. Each cap90; toy smoke is not quality evidence. All
actual smoke fits enter the total ledger. No broad repeated ML test matrix.

Deliver separate PR, explicit decision, compact JSON/CSV, Linux/PowerShell
commands, data-free evidence ZIP (summaries, predictions, source/recipes,
receipts, ledgers, environment, manifest; no raw fit/held/answers/models), and
integration request. No shared-file edit, default activation, native admission,
merge, platform submission or contest-score claim is authorized here.
