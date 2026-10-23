# Round 5 regression: isolated candidate handoff, no native admission

Base: `66e9481cb61cb83cfecab53be6984d7d2bf829b3`.
Family: `round5_regression_ensemble_v1`, version 1.
**Decision: retain the integrated `rescue_regression_v1` incumbent.** No new arm
passed the preregistered gates. E75 improved pooled development RMSE by 0.7233%
and won two of three transitions, but period-2 Residential SSE increased by
18.6023% across 6,449 rows (10,123,578.07 to 12,006,794.63; +1,883,216.56).
That material segment failure prevents recommendation. LGB had material
segment failures on periods 2 and 3 and stopped before confirmation period 4.
No threshold, weight or recipe was changed after those outcomes.

The candidate remains inactive. This PR does not add a scheduler/CLI hook,
change a default, replace the integrated rescue, or establish native admission.
No selected finalist was frozen for full TRAIN, and no new full-TRAIN refit or
known-period-5 compatibility evaluation was run. The final source/dependency
hashes, recipes and complete negative evidence are in
`round5_regression_results.md` and the accompanying machine-readable results.
This interface is retained for audit and reproduction, not activation.

## Candidate and frozen inputs

```python
from agent.round5_regression_candidate import make_round5_recipe, candidate_source

recipe = make_round5_recipe(
    period_column,
    selected_arm,                 # one preregistered fixed arm, never tuned here
    history_config=frozen_schema,
)
source = candidate_source()
# Host-owned freeze_candidate/source identity and run_candidate remain unchanged.
```

`fit_predict(X_fit, y_fit, X_predict, context)` returns
`{"predictions": finite_vector}`. The pure implementation is
`execute_round5_candidate(..., return_evidence=False)`. Its optional evidence
return adds component/phase timings, explicit calibration-position hashes and
base/mean/calibrated-history ablations; it does not expose fitting labels.
The standalone source writes the ordinary non-scoring evidence and ablation
sidecars. The runtime implementation reads no files, discovers no history
schema, requests no network and examines no task/path/ID names.

The frozen JSON includes all component recipes, constant weights, role-based
feature subsets, seed, internal split, calibration alignment, history schema
and unchanged history policy, resource shares and failure policy. It accepts
only the seven predefined new arms; incumbent I runs the unchanged
`rescue_regression_v1` separately. Canonical finite JSON equality validates
types and complete settings, so boolean/numeric substitutions, added fields
and altered component weights are rejected. Host-frozen source and dependency
hashes bind the existing fixed preprocessing/estimator defaults.

Eligibility requires a validated regression task with original-scale RMSE and
numeric vector predictions; exact feature roles covering the passed frames;
an explicitly declared numeric period column; a finite original monotonic
deadline; and the installed fixed CPU engines needed by that arm. Numeric-only
is a global model on declared numeric columns, with copied feature frames and
derived roles. It has no separate routing for cold rows or categories. Existing
native-category caps still apply. Incompatible schema, model, environment or
deadline fails the invocation before a partial result can be accepted.

## Data nesting and exact lifecycle

1. The owner supplies one permitted fit and its labels. Candidate-local
   calibration holds out that fit's last valid period; only strictly earlier
   rows train each internal model and its preprocessing.
2. Each component retains only copied raw data, selected tree count and
   calibration predictions. Its fitted internal model is not reused for a
   final fit. Numeric and full-feature canonical row order may differ, so
   calibration explicitly checks identical original `iloc` fit/held sets,
   sorts held positions and reindexes each prediction vector before mixing.
3. Fixed weights mix these aligned base predictions. The unchanged strict-past
   history encoder/19-option selection is calibrated on exactly this mixture
   and those internal held labels, using the original raw features.
4. After history selection is frozen, every component is built anew and fits
   all supplied fit rows. Fixed weights mix its prediction vector, followed by
   the already selected history adjustment. Prediction-batch contents never
   choose weights, statistics, schemas or routing.

This uses **legacy tree alignment** for all engines: calibration predicts at
best iteration, while a fresh full refit uses `max(50, best)`. There is no
`refit_count` recheck and no hidden correction of that existing mismatch.
Best/final counts are exported. A future alignment experiment would change
the algorithm and require new preregistration; it must not be slipped into
native integration. Single-period supplied fit uses the existing fixed 400
trees and no internal calibration.

History discovery belongs to the owner, once per transition on the permitted
host INNER-fit with the existing 8M/48 limits and `extended=False`. No discovery
is allowed in this entry point or on a larger direct/full fit. The exact same
frozen schema must be reused in the isolated development and eventual full
refit. If development used `None`, full TRAIN also uses `None`; schema absence
means the exact cold/history groups are not assessable. A known-period-5 key
must not be substituted. The permitted final fit-only last-period stopping
inside full TRAIN is not an external known-period-5 evaluation or permission
to select another recipe.

## One deadline and explicit failure

The original invocation deadline pays for host preparation, input copies,
internal preprocessing/training/prediction, alignment, history calibration,
fresh full refits, final prediction and output. The candidate reserves
`min(8 seconds, 5% of remaining time)` for history/output and divides the
remaining model window into fixed 100% or 50/50 component quotas.

Each quota counts the component's cumulative staging and refit wall time.
Every stage expires at the minimum of the original base-window end and its
own remaining quota. Idle time while another component runs does not become
training time; unused quota is not transferred. Components and invocations
run sequentially with the existing eight-thread CPU settings. A slow first
component cannot receive another full invocation allowance.

Any incomplete component, mismatch, timeout or nonfinite output raises. There
is no silent partial blend or changed-model fallback. The owner can then keep
its previously verified incumbent under its original remaining deadline.
Within a successfully completed algorithm, existing history fallbacks remain
explicit: no schema returns the base mixture; unavailable/insufficient internal
history calibration uses fixed mean history weight 0.5 on eligible rows.
Those are frozen algorithm branches, not recovery from a failed component.

The measurement observer persists each actual engine fit start before calling
the library and records completion/failure afterwards. `active_component()`
labels `internal_selection` or `full_refit`; the observer also records actual
row counts. Expected fit counts in successful algorithm evidence are labelled
as expected and never substitute for actual failed-attempt ledgers. Callback
deadlines and sampled memory limits are cooperative controls, not an OS memory
sandbox; unknown resource telemetry remains unknown.

## Measured cost and owner hook

The isolated harness charged 180 seconds per screen invocation and 360 seconds
per confirmation invocation, including all components. The measured source was
`c0ba45118c41dfa4b0ad0360f925874692dd3082`. These are direct supplied-fit
experiments, not timings of the complete adaptive scheduler. All eight
confirmation invocations below completed their actual started fits. LGB's
missing period-4 confirmation was a preregistered stop, not a successful
zero-cost run; its earlier 180-second screen does not substitute for it.

| Period / fit → predict rows | Arm | Actual completed fits | Host preparation, s | Through candidate completion, s | Post-prediction diagnostics, s | Whole invocation, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 / 2,117 → 9,058 | I | 1 | 0.359 | 2.791 | 1.069 | 3.860 |
| 2 / 2,117 → 9,058 | E75 | 2 | 0.355 | 3.354 | 1.599 | 4.954 |
| 2 / 2,117 → 9,058 | LGB | 1 | 0.370 | 2.341 | 1.390 | 3.731 |
| 3 / 11,175 → 10,879 | I | 2 | 0.596 | 4.275 | 1.676 | 5.952 |
| 3 / 11,175 → 10,879 | E75 | 4 | 0.751 | 5.620 | 2.561 | 8.182 |
| 3 / 11,175 → 10,879 | LGB | 2 | 0.752 | 3.540 | 2.164 | 5.704 |
| 4 / 22,054 → 12,946 | I | 2 | 0.787 | 5.272 | 2.356 | 7.628 |
| 4 / 22,054 → 12,946 | E75 | 4 | 1.134 | 7.661 | 3.853 | 11.514 |
| 4 / 22,054 → 12,946 | LGB | Not executed | — | — | — | — |

"Through candidate completion" already includes host preparation; whole time
also includes diagnostics and final bookkeeping. Do not add these nested
columns. The worker run directory's `full_refit` phase label means refitting
the **supplied development fit**, not a final full-TRAIN fit or native admission.

For E75 confirmation period 4, the internal fit had 11,175 rows and calibration
had 10,879; each fresh component refit used all 22,054 supplied rows. The actual
fit ledger orders CatBoost internal, LightGBM internal, CatBoost fresh refit,
LightGBM fresh refit. Component timings were:

| Component | Staging including internal block, s | Internal block, s | Actual internal engine fit, s | Fresh pipeline fit, s | Actual refit engine fit, s | Final pipeline prediction, s | Best / final trees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CatBoost, weight .25 | 1.707 | 1.061 | 0.761 | 0.551 | 0.274 | 0.171 | 66 / 66 |
| LightGBM, weight .75 | 1.380 | 0.700 | 0.344 | 0.465 | 0.167 | 0.209 | 31 / 50 |

These are nested timers, not additive phases: staging contains input validation,
canonical preparation and the complete internal block; pipeline fitting
includes preprocessing and the nested engine fit; pipeline prediction includes
feature transformation and engine prediction. Additional per-component input
copies took 0.010/0.008 seconds. The complete candidate implementation took
4.527 seconds, including alignment and the no-schema history path. There were
no calibration recheck fits. Full stage events, preprocessing transforms,
prediction timers, original deadlines and actual fit ledgers are retained in
the result JSON and the data-free evidence archive; CSV contains aggregate rows.

All development schemas were `None`. Exact cold/history gates remain unknown,
base/mean/calibrated outputs coincide, and these measurements do not validate
history admission. CPU time, peak RSS and cleanup telemetry were unavailable:
the native sampler reported `RuntimeError: process inspection unavailable`
with zero samples. The configured 8 GiB cap is not a measured peak or an
8 GiB resource-qualification pass.

| Cost boundary | Evidence status |
| --- | --- |
| Isolated supplied-fit screen/confirmation | Measured above and in final results; no synthetic time estimate substitutes |
| Fresh selected full-TRAIN refit | Not executed: no finalist passed promotion gates |
| Native host INNER phase | Not measured/admitted in this PR |
| Native outer admission phase | Not measured/admitted in this PR |
| Native selected full-refit phase | Not measured/admitted in this PR |

There is **no recommended host activation from this result**. If a future,
separately reviewed experiment earns admission, the hook belongs where the
existing regression candidate is proposed/frozen after ordinary eligibility
and allowed INNER-fit schema discovery. It should freeze the reviewed source
and exact recipe, submit it to the unchanged evaluation/admission/refit path,
and retain original phase deadlines and incumbent-on-failure behavior. It must
not insert independent per-engine or per-phase budgets, copy fitted models
between host phases, infer eligibility from public dataset names, or overwrite
the existing rescue recipe family. A separate optional flag and scheduling
decision belong to the host owner.

One substantive acceptance test for that future hook: under a single small
absolute deadline, make the second component exceed its quota after the first
component completed. Verify the actual fit ledger preserves the started work,
no partial-ensemble predictions are admitted, the original deadline is
unchanged, and the previously verified incumbent can still finalize within its
remaining reserve. Successful ordinary-path tests must additionally establish
aligned calibration, fresh full refits for both components and final source/
recipe/receipt/CSV identity.

No new native admission, known-period-5 compatibility check or leaderboard
result is claimed. Known Year5 was reserved for the captain's optional
post-freeze compatibility check and must not influence weights or promotion.
