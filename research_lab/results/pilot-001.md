# External benchmark laboratory report

Status: **incomplete**. Profile: `pilot`. Scope: paired_cpu.

Saved evidence was independently re-audited. This report starts no training or downloads.

Candidate wins: **0/4 planned pairs**; ties: 0; losses: 0. Provisional pairs: 0.

Positive signed difference means candidate better: incumbent − candidate for minimize, candidate − incumbent for maximize. Only two freshly verified, fully observed CPU arms count; tie tolerance: 1e-12. Provisional differences are separate and never wins. Different metrics are not averaged; budget tiers remain separate.

| Dataset / seed / tier | Metric / direction | Incumbent | Candidate | Signed difference | Provisional difference | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bank_marketing / 1729 / cpu-360s | LogLoss / minimize | 0.723130221 | — | — | — | censored |
| wdbc / 1729 / cpu-360s | ROC AUC / maximize | — | — | — | — | not_completed |
| wine_quality_red / 1729 / cpu-360s | RMSE / minimize | — | — | — | — | not_completed |
| auto_mpg / 1729 / cpu-360s | RMSE / minimize | — | — | — | — | not_completed |

| Case | Arm | Outcome / state | Worker s | Cleanup s | CSV valid | Outer + full refit | Cleanup complete | Resources | Reason |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| bank_marketing__1729 | incumbent | censored / censored | 269.172296 | 0.001051536 | yes | no | — | censored | — |
| bank_marketing__1729 | candidate | not_started / prepared | — | — | no | no | — | unavailable | — |
| wdbc__1729 | incumbent | not_started / prepared | — | — | no | no | — | unavailable | — |
| wdbc__1729 | candidate | not_started / prepared | — | — | no | no | — | unavailable | — |
| wine_quality_red__1729 | incumbent | not_started / prepared | — | — | no | no | — | unavailable | — |
| wine_quality_red__1729 | candidate | not_started / prepared | — | — | no | no | — | unavailable | — |
| auto_mpg__1729 | incumbent | not_started / prepared | — | — | no | no | — | unavailable | — |
| auto_mpg__1729 | candidate | not_started / prepared | — | — | no | no | — | unavailable | — |

| Case | Arm | UTC start | UTC finish | Actual API attempts / responses |
| --- | --- | --- | --- | --- |
| bank_marketing__1729 | incumbent | 2026-10-21T21:00:49.246046+00:00 | 2026-10-21T21:05:21.689039+00:00 | 0 / 0 |
| bank_marketing__1729 | candidate | — | — | — / — |
| wdbc__1729 | incumbent | — | — | — / — |
| wdbc__1729 | candidate | — | — | — / — |
| wine_quality_red__1729 | incumbent | — | — | — / — |
| wine_quality_red__1729 | candidate | — | — | — / — |
| auto_mpg__1729 | incumbent | — | — | — / — |
| auto_mpg__1729 | candidate | — | — | — / — |

All pair outcomes: win=0, tie=0, loss=0, provisional_resource_observation=0, failed=0, censored=1, integrity_failed=0, skipped=0, not_completed=3, metric_contract_mismatch=0, fixture_smoke=0, self_check=0, missing=0.

Actual report API attempts / responses: 0 / 0; verified report coverage 1/1 started arms. Missing observations remain unknown.

Incumbent: `a45861d7e5588288db66f9ae088a9726f9597370`; candidate: `3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee`.

Plan SHA-256: `5fcc96615b245777aa43d0062d602405a59b1378a8e134998b0ee7c47e50b159`. Manifest SHA-256: `c653c841ee2e65e9019d3d914659addadd299ed0a38e18e3bd0179363eaf7f55`. Method SHA-256: `84ab410c0a14bc78572c2a7cc5447eee45ca361d30da4485e92bd68cabc12a8a`.

Evidence directory: `/workspace/scratch/de9281f42bd1/alfa-external-benchmark/data/lab-runs/pilot-001`. JSON and Markdown file hashes are saved in their `.sha256` sidecars.

Windows-native execution and GPU experiments are not established by this CPU report. Synthetic fixture smoke is mechanism evidence only. Local quality does not establish a contest-score improvement.
