# Classification rescue v2: frozen development plan

Recorded before any model fitting in this work, 2026-10-22.

- Incumbent: `d05572846369f3b1e9cfa33f4008dda028db4fe0`, freshly fetched `origin/main`; identical to the assignment baseline.
- Branch: `work/rescue-classification-v2`; independent clone, with an additional detached unchanged-main worktree for the native incumbent.
- Official public data: downloaded and checked by `python prepare_data.py classification`. Original CSV files remain unchanged. Competition TEST has no labels and is never a development target.
- Python 3.12; modelling packages pinned to `requirements.platform.txt`. No dependency installation in candidate code. AutoGluon is absent and not part of this bounded study.
- CPU only, 8 threads. Host cgroup allows 8 CPUs and **8 GiB RAM**, not the official 32 GB. Child tree RSS limit: 6 GiB, sampled by the existing supervisor. This is not evidence of a 32 GB platform run.
- API mode: offline, no client or API calls. Native reports must confirm zero attempts/responses.

## Observations and hypotheses

The freshly loaded TRAIN has 87,670 rows, 42 features (3 categorical), no missing feature values, and 10 classes. `Worms` has only 65 rows versus 28,000 `Normal`. Raw feature hashing excluding ID/target finds 54,228 groups, 5,816 duplicate groups containing 39,258 rows, and 1,458 conflicting-label groups containing 14,829 rows. Grouped validation is essential; these counts do not establish a Macro F1 ceiling.

Test three explanations: (1) rare-class decisions need moderate training weights rather than blanket balancing; (2) a longer single boosting model may improve over short scheduler trials; (3) a regularized XGBoost model may be competitive once it actually reaches training. Missing indicators are not motivated by the observed data. No feature engineering or post-hoc label weighting is planned in the initial screen.

## Fixed comparison and limits

1. Measure unchanged main through `solution.py --mode offline --research-search adaptive --time-budget 900` on a derived task whose TRAIN/TEST are the grouped seed-42 fit/development partitions. Held-out answers stay outside the candidate task. Save the actual published predictions and complete native reports. This measures the complete incumbent scheduler, not a single recipe.
2. Freeze grouped dev splits for seeds **42, 43, 44**, using the existing `agent.splits.split_rows` policy, changing only its external `offline_baseline.SEED` during split construction and restoring immediately. Internal/model seed remains 42. Group overlap must be zero on every split. No seed substitutions after results.
3. Initial screening: five fixed recipes on seed 42: CatBoost depth 6 unweighted; the same with `SqrtBalanced`; LightGBM with 31 leaves; existing XGBoost `depth6_v1`; existing XGBoost `depth4_regularized_v1`. Each uses existing fit-only early stopping and fresh full-fit refit, at most 2 model fits each: **10 fits total**. No OOF or ensemble in this first screen. Per-invocation timeout 300 seconds, maximum 2,000 iterations, stopping patience 50. Reserve refit inside the existing candidate policy.
4. Advance exactly the two highest successful Macro F1 recipes to seeds 43 and 44. Publish all planned outcomes, including timeouts/errors and class metrics. Ties follow frozen candidate order. These are development slices, not independent tests.
5. Useful gain threshold fixed now: **+0.005 absolute mean paired Macro F1**. A drop worse than 0.01 Macro F1 on any repeated slice or 0.05 F1 in a class requires investigation and prevents a default recommendation. Small-class support and recall must accompany F1. These are engineering selection thresholds, not statistical guarantees.
6. Refit the selected candidate on all public TRAIN and validate a newly published CSV locally. Do not submit it to the platform. This verifies isolated execution only.

The two isolated finalists can be compared pairwise across all three slices. A complete-scheduler incumbent result available only on seed 42 must not be misrepresented as a three-slice paired improvement over main. Existing recipes being measured is not itself a production algorithm change. If no useful stable improvement is demonstrated, retain incumbent and report negative evidence.

Protocol clarification recorded while the first native baseline is still running, before observing any external Macro F1: after its publication, freeze the **actual published incumbent code and recipe** and replay that fixed control on seeds 43/44 through the same isolated worker at 300 seconds. The seed-42 control remains the native published result with identical fit/predict IDs. This supplies a three-slice comparison against the fixed incumbent recipe, not a rerun of the whole scheduler. It adds no challenger or tuning choice. Retain original elapsed times and unequal selection-budget provenance; same-scheduler candidate-off/on evidence remains pending integration.

## Integration boundary

Only assigned classification/experiment/test/document paths may change. No shared scheduler/router, requirements or API changes. First PR may be `integration_pending`. After the integrator supplies an integration SHA, compare candidate-off/on arms at that same SHA with identical scheduler configuration and 900 seconds, then one longer budget only if evidence warrants it. Until then no claim of improved native competition routing or platform score is permitted.

## Observed resource limitation (after the first native run)

The existing supervisor reports `available=false`, `peak_rss_bytes=null`, `sample_count=0`, and `RuntimeError: process inspection unavailable` because this execution environment's process namespace differs from its mounted process inventory. The requested 6 GiB sampled process-tree RSS limit therefore **was not enforced**. The 8 GiB container cgroup remains the actual shared memory bound. Do not replace the missing RSS with the configured limit, do not claim verified child-tree cleanup, and do not claim competition resource qualification. Finite child/solver wall-clock deadlines remain active. This limitation applies to the following isolated runs if their existing monitor returns the same unavailable result; retain those nulls in every outcome.
