# Native queue f6b3e851: findings and follow-up

Evidence: user-exported `alfa-native-f6b3e851.zip`, queue
`f6b3e8514bd54cbbb6d00c3c147bc278`, source release
`5672b5f655eba90da1aefdf9c7b11e7c934ceba0`. Completed 2026-10-23
18:25 UTC (21:25 Moscow), after approximately 85 minutes.

| Job | Published outer metric | Native process time |
| --- | --- | --- |
| regression_off | RMSE 44.56965721017985 | 14m 33s |
| regression_on | RMSE 39.47145825573459 | 16m 35s |
| classification | Macro F1 0.6257309911277048 | 53m 48s |

All three root processes completed, input/source seals and selected execution
receipts passed, cleanup was observed, and API attempts were zero. The Qwen
model name in metadata does not imply an API call. These are known public TRAIN
development scores: outer scores participated in selection, so they are not an
unbiased estimate of hidden TEST quality. The ON/OFF regression difference is
5.0982 RMSE on the same development split, not a promised leaderboard gain.

## Search failure to use the available budget

Each native search stopped with `local_research.reason=trial_limit`, exactly
64 verified ordinary trials, zero ordinary improvements, and zero failed
ordinary trials. Time remaining was about 9708 seconds OFF, 9718 ON, and 7319
for classification. Startup recipes consumed 14–16 slots before the adaptive
proposer; the remaining attempts were spread across five model families.

The promotion rule needs three measured trials at a family/resource level to
promote one candidate. No run reached the top resource level. CatBoost and
LightGBM received only two middle-level trials each, leaving the 1200-tree
level unreachable within the observed attempt allocation.

The classification winner stayed the startup LightGBM recipe: 500 trees,
63 leaves, minimum child size 40, learning rate .05, balanced class weights.
Its inner F1 was .6342936339; the best adaptive inner F1 was .6198618094.
Inner and outer scores use different rows and must not be directly ranked.

## Other actionable findings

- AutoGluon was skipped with `autogluon_unavailable` in all jobs. It is present
  in the platform requirements but absent from the compact local install.
- Native ensemble was skipped with `ordinary_measured_finalist_required`.
  Its admission requires an individual improvement, excluding potentially
  complementary weaker models. App weighted blending is a separate experiment;
  it does not silently alter this native admission rule.
- Three startup CatBoost refinements in classification hit their individual
  timeouts. The accepted LightGBM checkpoint survived; the complete queue did
  not fail.
- Feature transformation emitted 3276 pandas fragmentation warnings from
  repeated column insertion. Building the output frame once removes this
  avoidable overhead without changing feature values.
- Available system RAM at queue preparation was about 3.55 GB. No OOM was
  observed, but larger experiments should be run with other memory-heavy work
  closed; physical RAM alone is not the available training budget.

## Follow-up protocol

Use the explicit `adaptive-budget` native mode for new workstation queues,
retaining the original deadline, finalization reserves, integrity checks, and
256-trial safety ceiling. Existing `adaptive` remains a reproducible control.
Compare identical source/data/split settings and retain stop reasons, resource
levels, inner improvements, and the final published result. Increasing the
ceiling is an opportunity to search deeper, not evidence of better quality.

Ordinary app AutoML additionally tests inner-selected weighted ensembles.
TabM is a separate optional CPU/CUDA experiment trained from scratch, requiring
its own runtime and measured comparison before promotion into submission logic.
