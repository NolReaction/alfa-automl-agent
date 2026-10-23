# Round6: preregistered fit-only residual calibration

Declared 2026-10-23 MSK before any new model fit, including smoke/probes.
Branch `work/round6-regression-calibration-v1`; current origin/main base
`114fc2c18595898e0f6ec099e42795eec4af0be9`, tree
`0ee506c852646704037620759dc35ac0fa688072`. This includes the accepted inactive
Round5 study. The 17 previously frozen legacy dependency files match their
Round5 SHA256 values. New app/readiness commits do not change those bytes.
No source or measured-base change is allowed during this series.

Before the first model fit, plan and implementation must be committed clean,
published, and verified to have identical local/published Git trees. Record
both SHAs when publication metadata changes commit identity. Input hashes,
dependencies, exact recipes and numeric constants are frozen in the paired
`round6_regression_plan.json`; source snapshots bind implementation details.

## Fixed hypotheses and arms

| Order | Arm | Correction before unchanged history |
| --- | --- | --- |
| 1 | I | Unmodified integrated CatBoost rescue-v1 |
| 2 | S20 | 0.2 times mean residual on A, clipped |
| 3 | S50 | 0.5 times mean residual on A, clipped |
| 4 | A20 | Ridge affine residual correction, penalty 4, clipped |

Exactly these four screen arms. Same original-scale RMSE CatBoost base:
depth6, learning_rate0.05, l2_leaf_reg5, maximum2000 trees, patience100,
single-period fallback400, seed42, CPU8. Existing estimator/preprocessing
defaults remain bound by source hashes. No engine, target transform, seed,
weight grid, category-specific rule or frequency feature is introduced.
A20 does not multiply the ridge result by an additional 0.2: its identity
shrinkage is the explicitly specified penalty 4.

## Three nesting levels and deterministic A/B groups

For external development period t, supplied fit contains strictly earlier
periods. History discovery uses the host INNER-fit obtained by excluding the
latest supplied-fit period. Separately, the candidate's calibration is the
last period within the supplied fit; its base-training prefix contains only
earlier supplied-fit periods. The candidate receives no external held labels,
host-held labels outside its supplied fit, original TEST labels or files.

Before model fitting, divide this internal calibration into A/B by raw-feature
group. Use all declared raw features, excluding ID, target and row index;
lexicographically sorted feature names and a canonical typed scalar encoding
with explicit missing values. SHA256 includes seed42. The unsigned digest
modulo100 is assigned to A when below70, otherwise B. Identical raw groups
stay together; row order and labels cannot select their side. The exact
encoding is frozen in the pure helper source and recipe. Export counts and
aggregate membership hashes, never raw group strings/values.

Require at least100 rows each in A and B. No calibration, insufficient support,
invalid statistics or a nonrepresentable diagnostic partition returns the
unchanged incumbent algorithm with an explicit reason. Absence/support can
be checked before trial fits; no models are trained to invent calibration.

## Aligned trees, correction and one B decision

1. The unchanged internal base helper fits only the strict earlier prefix and
   uses **A labels only** for early stopping/tree selection. B labels never
   select trees for the correction path.
2. Always pay one fresh prefix recheck at `max(50,best_selected_on_A)` and
   predict both A and B at this exact count. This paid fit is necessary for
   this reuse path because the legacy helper releases its fitted model and
   returns only A predictions. It also eliminates the old best-vs-min50
   calibration mismatch. Prefix preprocessing is fitted afresh on prefix
   rows only; no fitted model is carried into a full supplied-fit refit.
3. Fit the correction once using A. S20/S50 use
   `d=factor*mean(y_A-p_A)`, clipped to `±0.1*std(y_A)`. A20 uses
   `z=(p-mean(p_A))/max(std(p_A),eps)` and minimizes
   `mean((y-p-b-a*z)^2)+4*(b*b+a*a)`. Solve
   `(X.T X/n+4I) theta=X.T residual/n`. Both intercept and normalized slope
   are penalized. Clip the resulting multiplicative slope to [0.9,1.1] and
   correction at mean(p_A) to `±0.1*std(y_A)`.
4. Freeze `ddof=0`, `eps=1e-12`. Nonfinite values, absent required spread,
   degeneracy or zero/nonfinite target standard deviation cause identity.
   B evaluates the frozen A parameters exactly once. Corrected B SSE must
   improve identity by more than `1e-12*max(1,identity_B_SSE)`; a numerical
   tie favors identity. This stricter tie rule satisfies the required
   non-increasing SSE gate and avoids accepting ineffective corrections.
5. Apply the unchanged Round5 significant history/category/novelty veto
   thresholds on B. Category levels/novelty and all fit-derived boundaries
   come from the permitted earlier prefix. Unsupported segments are unknown,
   not passed. Schema=None makes history unknown. Preserve old history-group
   membership; if a supplied schema makes its old partition invalid, reject
   the correction rather than redefine or waive the gate.
6. Do not refit A parameters on A+B or compare corrections inside one arm.
   After acceptance freeze this policy; calibrate unchanged history on the
   corrected aligned internal predictions. History retains its existing
   allowed fit-only selection, including internal calibration labels; this
   does not reopen correction fitting or tree selection.
7. Freshly fit the base on all supplied-fit rows at the selected aligned tree
   count, apply the frozen correction, then the frozen unchanged history
   policy. Record `identity` (aligned uncorrected base), `base` (corrected
   pre-history), and `calibrated` (post-history) ablations.

**Exact fallback and scope of the B restriction.** Legacy I selects trees
using its entire latest supplied-fit period. Therefore a literal ban on B
labels even inside I would contradict the requested exact incumbent fallback.
The frozen interpretation is: B never selects correction-path trees or A
parameters; **only after correction rejection**, a separate call to unchanged
I retains its existing selection/history contract. It receives the original
remaining deadline, never a reset. An incomplete fallback fails the whole
invocation, with no partial prediction or altered A-only base called I.
Single-period/insufficient-support branches call I directly. Exact fallback
means the unchanged algorithm; a binding timer does not justify claiming
unconditional numerical equality to a separate fresh-budget run.

Accepted-path A-only stopping and aligned recheck can themselves change the
base relative to I. Keep that effect visible using the aligned identity
ablation and separately measured I; do not attribute every accepted-path
gain to correction. The algorithm has three expected fits when accepted
(selection, prefix recheck, fresh full supplied-fit model), usually four
after a trial rejection (selection, recheck, legacy internal and full models),
and only I's one/two fits on early fallback. These are CatBoost estimator
fits. Also observe the scalar/ridge correction fit as a separate analytic
parameter-fit operation (normally one per trial, zero on early fallback),
with persisted starts/completions and time. Report engine fits, analytic fits
and their combined total; no learned fitting operation is free. Actual
observed starts and completions, including failures, are authoritative.

## Fixed development queue and gates

Screen all four arms on period4, 180 seconds per entire invocation. Promote
at most two with >=0.2% lower RMSE than fresh I. Rank by RMSE; differences
below0.2% favor fewer actual physical fits and then fixed arm order.
Confirmation uses fresh equal-cap I plus those finalists on periods2,3,4,
360 seconds each. Screen/180 never substitutes for confirmation/360.

Qualification requires all three completed paired transitions, at least two
practical RMSE gains >=0.2%, pooled RMSE gain >=0.2%, and **no individual
RMSE loss above0.2%**. This explicitly tightens Round5's 5% fold limit to the
Round6 instruction's 0.2%; other significant-segment thresholds are retained.

A significant segment has >=max(100,ceil(1% held rows)) and control SSE>=1%
of total control SSE. History-group SSE deterioration >10% is a veto when
assessable. A category/novelty group veto requires >10% SSE deterioration
and absolute increase >1% of the total control SSE. Preserve all small-group
rows and absolute SSE; overlapping partitions cannot be summed.

A material failure is RMSE loss >=0.2%, a significant-segment veto or a
failed invocation. After two failures on distinct confirmation periods stop
the finalist. Incomplete gates stay unknown and cannot qualify. Preserve the
boundary `>=0.002` for stopping from Round5; acceptance's maximum loss is
`<=0.002`. No recipe, seed, cap or new arm may follow a disappointing score.
At most one documented technical retry uses identical source/recipe/seed/cap
in a new folder; preserve the first failed attempt. Audit-only fixes preserve
original reports and issue an explicit supplement without model reruns.

Use one allowed history discovery per transition, 8M/48, extended=False,
on the same host INNER-fit, freezing its exact schema for every arm. No known
key substitution. If every development schema is None, freeze None. If
schemas differ, do not qualify an untested configuration; retain incumbent
pending captain review. History/cold unknown can support at most a frozen-None
base-only development candidate, never history/native admission.

**Do not run Year5 or new full TRAIN/original TEST in this round.** A qualified
candidate may be frozen for captain review; compatibility and native admission
require that later decision. No qualified candidate means retain incumbent.
These exposed temporal slices are development evidence, not blind evaluation.

## One budget and deadline admission

All work is sequential, CPU8, API0, GPU0. Global heavy ceiling7200 seconds
includes preparation, discovery, diagnostics, smoke, validation, preflight
failures, failed fits and any permitted retry. Reserve before starting work
and preserve a single serial ledger. A persistent advisory guard protects
ledger updates; running reservations prohibit simultaneous invocations.
Preflight failures have their own terminal records and measured elapsed time.

One original invocation deadline covers all host preparation, internal
staging, model fits, alignment, correction gating, fresh fit/fallback,
history, prediction, diagnostics and success checks. Reserve
min(8seconds,5% initial remaining) for history/output; trial work gets a fixed
40% of the initial remaining model window. The accepted fresh fit or rejected
incumbent fallback uses only the remaining original allowance.

Fix the identified Round5 P2 issue only in this new harness: check the
original deadline **after diagnostics and source/receipt/output/ledger
verification, immediately before success publication**, with fresh elapsed
time. Over-cap results cannot be completed/admitted. A reader independently
rejects completed summaries with nonfinite/negative or over-cap elapsed time.
Test both writer and reader with cap180 and fake elapsed201.

Fit starts are persisted before each actual engine call, with engine, phase,
rows, deadline, start/completion/failure and elapsed time. Nested component,
preprocessing and observer timers are not additive; host wall is authoritative.
CPU seconds, peak RSS and process cleanup are unknown if process inventory is
unavailable. Configured8GiB is not a measured resource qualification.

Existing Python3.12.14 environment is reused. Core pins verified without fits:
numpy1.26.4, scipy1.12.0, pandas2.2.3, sklearn1.5.2, CatBoost1.2.8,
LightGBM4.6.0 (dependency only), psutil6.1.1. Full environment is exported.
Original input hashes match Round5; dataset names appear only in developer
manifests, never candidate routing. Freeze exact permitted split identities.

## Checks and deliverables

Model-free tests cover A/B groups and label boundaries, prefix preprocessing,
tree alignment, no mutation, roles/frozen recipes, exact single-period I,
constant/nonfinite/support fallbacks, clipping/ties/veto and unknown segments,
source identities, deadline writer/reader guards and actual fit accounting.
Exactly two real small S20 flat-ZIP smoke invocations outside checkout use
the full frozen recipe, cap90 each, aggregate reservation195. A synthetic
three-period fixture supplies >=100 rows in A and B. Reordering training/query
rows and adding unrelated extreme query rows must preserve original
predictions. No agent directory is assumed. All actual smoke fits count;
this is contract validation, not quality evidence.

Independently recompute RMSE/MAE/SSE/segments from every completed output.
Provide a correction disposition table: applied, B rejected, insufficient
support, no calibration, with explicit reasons and counts. Preserve all
failed/rejected results and all ablations. Export data-free ZIP with source,
recipes, receipts, fit ledgers, environment, predictions and SHA manifest;
exclude original TRAIN/TEST, answers, raw groups and fitted models.

One unmerged PR supplies base/measured/published/head SHAs and verified tree
equality, results MD plus compact JSON/CSV, Linux and Windows PowerShell
commands, real measured costs and a clear all-gates decision. A measured-cost
integration request is prepared only for a qualified variant. Runtime stays
inactive; no shared files, platform submission or native-admission claim.
