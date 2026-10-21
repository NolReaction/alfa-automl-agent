# History runtime v1: bounded developer evidence

Native frozen v3 local_only, client=None, fixed history weight 0.5; API requests 0. Control and history have equal per-run walltime budgets. History remains opt-in.

| Stage | Case | Control RMSE | History RMSE | Gain | Same baseline | History result |
|---|---|---:|---:|---:|:---:|---|
| development | entity_signal_131 | 0.455631207 | 0.363973575 | 0.091657632 | yes | verified: improved |
| development | no_history_coverage_131 | 0.191785815 | 0.191785815 | 0.000000000 | yes | verified: not_better |
| development | colliding_keys_131 | 0.046740247 | 0.046740247 | 0.000000000 | yes | verified: not_better |
| development | iid_regression_131 | 0.158181044 | 0.158181044 | 0.000000000 | yes | skipped: time_validation_required |
| confirmation | entity_signal_2203 | 0.445192825 | 0.365332279 | 0.079860546 | yes | verified: improved |
| confirmation | entity_signal_4001 | 0.420301501 | 0.380022835 | 0.040278666 | yes | verified: improved |
| confirmation | colliding_keys_2203 | 0.058738512 | 0.058738512 | 0.000000000 | yes | verified: not_better |
| confirmation | colliding_keys_4001 | 0.054897721 | 0.054897721 | 0.000000000 | yes | verified: not_better |
| public_sanity | known_public_year5_0 | 47.678458307 | 45.944664105 | 1.733794201 | yes | verified: improved |

Main results: 16/16 trials re-verified; 3 paired improvements, 5 exact ties, 0 regressions. All 8/8 startup baseline hashes match. History was requested in 8 runs: 7 inner trials verified and 1 policy skip. Active coordinator time: 406.476s.

Main method SHA256: `66953779a31667094f77b700cff0ab285e900e4225c72f6968100b4ea3f2d729`.

The main 16 runs used 90s each and a shared 1800s active coordinator budget. Development used four cases at seed131; confirmation used signal/collision at seeds2203/4001. Its method was sealed after development finished and before confirmation started. The public TRAIN sanity, when present, has 300s per arm and a separate 700s coordinator budget.

Re-audit checks frozen source/input/split hashes, exact fit/predict rows, inner history feedback, outer selection, complete refit, nonce/code/recipe/output receipts, published CSV and checkpoint. The compact JSON records exact hashes and counters; raw sealed manifests retain full file inventories.

Compare history quality only where its trial is verified. An eligible skipped/failed trial can still produce a valid end-to-end fallback. Recipe mismatch prevents a pure history attribution; even matching recipes can have different local admissions under the same deadline. Synthetic cases test deliberately specified mechanisms. Public TRAIN is known and previously studied. Neither establishes private score.

Observed local cgroup: 8 CPU, 8GiB. Child cap requested: 6144MiB; process inventory unavailable, so peak RSS and cap enforcement are not verified. Platform provides 32GiB; this is no platform qualification.

Runtime local 782d738a = remote 7cfa8c98 (same tree aef1d8f3); source SHA256 `28f7a51ebbc9833367a2de21a64b653b7e2266a068d2926f14589f9f36197c2e`. Harness local 41ba4ecf = remote c5f12722 (tree aa5875d7). Full refs are in the JSON.

Known public TRAIN year5 sanity: 47.678458307 to 45.944664105 RMSE (3.636% lower); 202.334s active coordinator time. Both arms verified four local trials with no accepted local improvements, and their startup baseline hashes match. The explicit six-key configuration is confined to this experiment. This case was previously studied and is not independent confirmation.

Public method SHA256: `fd6e733167e4faaacec47f88a6c699243d8eae1eb452d713ea2a3fa22855c4f7`.

Full machine-readable evidence: [history_runtime_results.json](history_runtime_results.json). Frozen execution protocol: [method at c5f1272](https://github.com/NolReaction/alfa-automl-agent/blob/c5f127223dde0eb935d021a62e2d2fdbb5fbcc55/docs/history_runtime_experiments.md).

Native CI for the final runtime/harness head passed on both Windows and Linux, for both push and PR: **747 core tests + 227 application tests per job**, launcher verification and temporary flat archive build. [Push run](https://github.com/NolReaction/alfa-automl-agent/actions/runs/39606295762), [PR run](https://github.com/NolReaction/alfa-automl-agent/actions/runs/39606302370). Fault-injection tests verify publication of the real baseline after history full-refit failure; outer-target intervention leaves research selection unchanged.

An independent read-only reviewer re-audited all 18 quality runs, the separate 2-run smoke and all five original public input hashes. The preserved raw evidence archive contains 6,484 files (15,877,113 compressed bytes, identical content stored as standard tar hardlinks); SHA256 `360c136df61291f183ad59054889d933af19706526ed12692ba24fd2279c7110`. The saved manifests retain original absolute paths; portable reruns use the frozen protocol rather than rewriting their seals.
