# Round 4 Qwen: bounded evidence, measured protocol, negative stopping result

The bounded INNER evidence/helper and a pure next-action policy are ready for host-owner review. The exact-repeat loop used fewer requests and one fewer physical fit, with the same accepted output. **The unconditional two-nontrial stop is not recommended as a default:** it suppressed a later useful candidate in the preregistered counterexample. Thresholds were not retuned after observing results. No production policy, host, baseline, model, dependencies or contest release was changed.

## Provenance and activation

- Fixed base: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`. A later `git fetch origin` still found this exact main; comparison base was never moved.
- Local preregistration: `aaa9a1e953f2706e9cfc5a8c93e8ada591897df8`, before new candidate screening.
- Helper implementation: `8d184e5fc633584e6289249a65e484d8e21ef6e2`.
- Measured implementation HEAD: `cd8161a5a80434e7b3254a2e6697740b05baa1f6`; clean worktree before/after, source unchanged during all ten arms.
- Equivalent published implementation HEAD: `f4803483f3f78b8777089ddeceada6e243996b5e`. Its Git tree is exactly the tested local tree `7efdf4b166d4c44aed0f902dd7ca43533cc996b9`. Connector-created commit metadata yields different commit IDs; code bytes are identical.
- Published preregistration/helper commits: `30be1749c26cef8b9bc14d23c9a9f7ba2f337b80` / `149a75dc2d59a7731e622c457782a4c17e2e2dd6`. Final report/results commit and exact delivery HEAD are visible in the PR; they add evidence only after measurements.
- Policy version: `round4_inner_action_v1`. Activation: **test harness only, before native guards**. Production: inactive.
- Measured interval: 2026-10-22T21:40:46.472389+00:00–2026-10-22T21:41:46.745148+00:00 UTC; 2026-10-23 00:40:46–00:41:46 MSK.
- Source SHA-256: `53f7672c8d19418e410a3e32ec9269811c244efc68862fb404c4344adf9f28be`. Harness SHA-256: `5cb75a0a9f902de21c00f3b445e710ddf41ea5dbe3b76f9b7a61aac6c4234e16`.

The machine-readable [JSON](round4_llm_results.json) and [CSV](round4_llm_results.csv) contain exact source/input/split/script/code/recipe hashes, environment versions, native receipt-derived milestones and counters. Source inventory contains repository-relative filenames and hashes only. Public exports exclude prompts, tool arguments/source, native logs, request IDs, CSV contents, secrets and personal paths. Native logs remain local.

## What was compared

Five cases, one fixed policy, two arms in sequence, no network, at most four mock requests and 90 seconds per arm. These are **scripted mock responses**, not recorded live-model replay. Replies ignore the new hints; therefore any effect on actual Qwen hypothesis quality is unmeasured.

`main_context` uses the unchanged e62fea1 request policy. `round4_context` adds the proposed evidence/policy through a thin test-only adapter before the original request function. The same original messages, allowed input files, data, split, scripted reply sequence and caps are available to each arm; new inline aggregates expose information the old arm could inspect in those files. No client wrapper changes the transmitted payload after the guard. Hash comparisons confirmed that every sent message payload was the exact guard-checked payload.

Both arms use the same cheap **fixed mean baseline**, with empty local recipe plans; candidate execution, scoring, freeze/admission, outer checks, full refit and CSV validation use the real unchanged host. Baseline training is charged within each arm. This is a protocol comparison against main's request machinery, **not a repeat of main's full baseline portfolio or a production quality comparison**. Production offline vs Qwen quality with identical total budgets and hidden answers remains a captain-owned unmeasured experiment.

Fixture: 100 TRAIN rows and 2 TEST rows; deterministic synthetic regression. Outer split: random, seed 42, 80/20 rows. INNER within outer fit: random, seed 42, 64/16. Input SHA-256 `3a3fe8c54d06f6bc1e9f1795e9f19d9d03058bb7d70c8b72c7b49f9030314a97`; exact inner+outer split hash `3c49a48e70b0587e49ac32192bed629aace2d0986c91aeeb1bf5606ae4e33ed0`. All five pairs have matching nonnull input/allowed-input/split/script/limit identities and unchanged input bytes. Outer scores are summarized only after dialogue closure, never supplied to the helper.

## Paired measurements

Every arrow is main → proposed. Physical candidate fits below include INNER, outer and full refit; they exclude baseline fits. First-INNER time starts at host creation; total arm wall time also includes fixture setup and public-summary validation. A single timing sample is descriptive, not a latency benchmark.

| Case | Requests | Unique verified INNER | Duplicate verified INNER | Candidate physical fits | First verified INNER, s | Whole arm, s |
| --- | --- | --- | --- | --- | --- | --- |
| useful | 2 → 2 | 1 → 1 | 0 → 0 | 3 → 3 | 2.389 → 2.423 | 7.130 → 7.291 |
| analysis_only | 3 → 2 | 0 → 0 | 0 → 0 | 0 → 0 | null → null | 3.601 → 3.702 |
| no_training | 2 → 2 | 1 → 1 | 0 → 0 | 0 → 0 | 2.489 → 2.349 | 4.870 → 4.710 |
| exact_duplicate | 4 → 2 | 1 → 1 | 2 → 1 | 5 → 4 | 2.460 → 2.594 | 9.473 → 8.354 |
| late_useful | 4 → 2 | 1 → 0 | 0 → 0 | 3 → 0 | 2.501 → null | 7.467 → 3.646 |

| Case | Total request bytes | Native estimated token reservations | Valid recipe fraction | INNER-improving fraction |
| --- | --- | --- | --- | --- |
| useful | 15942 → 22480 | 21406 → 23585 | 1.000 → 1.000 | 1.000 → 1.000 |
| analysis_only | 20019 → 19227 | 30810 → 22501 | null → null | null → null |
| no_training | 15334 → 21864 | 21204 → 23380 | 1.000 → 1.000 | 0.000 → 0.000 |
| exact_duplicate | 43136 → 22480 | 46563 → 23586 | 1.000 → 1.000 | 0.333 → 0.500 |
| late_useful | 30105 → 19231 | 42219 → 22503 | 1.000 → null | 1.000 → null |

Valid recipe fraction means host-verified INNER trials per executed tool experiment, not syntax-only validity. INNER-improving fraction is accepted INNER updates per verified trial; it is not independent final quality. Undefined fractions are null with reasons in JSON.

- **Useful:** both arms publish the same generated linear candidate, with matching source/recipe identities through verified INNER, freeze, outer admission, full refit and fresh CSV. INNER RMSE is approximately 2.404e-14 and known synthetic outer RMSE 7.229e-14. Extra evidence increases payload by 6,538 bytes; earlier usefulness is not demonstrated (2.389 → 2.423 seconds).
- **Analysis only:** 3 → 2 requests, no candidate in either arm, baseline finalization preserved. Whole-arm time is slightly higher despite one fewer zero-latency mock reply; no real transport-latency savings were measured.
- **Exact duplicate:** 4 → 2 requests, duplicate INNER trials 2 → 1, candidate physical fits 5 → 4 (all fixture fits including baseline 8 → 7). The same unique candidate is still published. This stops after a repeat is observed; it does not prevent the first duplicate execution or implement host deduplication before training.
- **No training:** both arms execute one zero-prediction candidate and verify its INNER predictions, but candidate physical fits are **0**. All three physical fits belong to the baseline. The generated candidate is not accepted or published. Verified predictions, successful processes and actual training are separate milestones.
- **Late useful, negative:** old policy reaches the valid third reply and publishes the linear candidate; proposed policy stops after the two analysis replies, executes no candidate and publishes the mean baseline. Known synthetic final outer RMSE changes from about 7.229e-14 to **86.9696**. The smaller request/fit counts represent a lost opportunity. No threshold was relaxed to hide it.

Across ten arms: zero harness failures, zero network requests, ten valid fresh finalizations and ten guard/payload matches. Request totals are 15 → 10; physical-fit totals 26 → 22. These totals combine an avoided duplicate with a suppressed useful candidate and **must not be presented as a net quality win**. Provider usage and currency cost remain null: every mock response omits usage. Native token reservations use the existing 8192 completion reserve and are distinct from transport byte estimates and actual usage.

Physical-fit counts use completion markers placed after the known fixture's mean fit or `numpy.polyfit`; the untrained zero function emits none. This measures these fixed sources only. It cannot infer arbitrary generated code's internal fits. Fit-only time is null because execution wall time includes startup, preparation and prediction. Peak RSS and CPU seconds are null: native process inspection returned unavailable, sample_count=0 (`RuntimeError: process inspection unavailable`). The 28 GiB native candidate memory cap is not a measured peak and does not qualify the full 32 GB/10,500-second official runtime.

## Risk checks and reproducibility

**71 targeted tests passed in 32.938 seconds:** 24 new evidence/policy/host/adapter tests plus 47 existing gateway/diagnostic/resilience/smoke checks. After that combined run, only split/provenance metadata capture was added to the harness; the final clean implementation passed the complete ten-arm CLI above. Existing smoke fixtures already passed on the fixed base; the prior-round migration issue was not reopened.

The checks include an actual outer-target intervention under a frozen split (INNER evidence unchanged), exact probability labels/order, recipe-hash tampering, immutable SYSTEM_V3/tool pairs, invalid scope with zero sends and baseline finalization, actual invalid child labels/probabilities, truncated response with no tool execution, typed HTTP 504 with exhausted original retry time, verified INNER losing outer without forced candidate refit, local winner retaining local provenance, unknown usage, and import from a flat ZIP without source-checkout access.

Actual Linux commands, from this worktree:

```bash
/workspace/scratch/b69ed5d5758a/qwen-test-env/bin/python -m unittest tests.test_round4_llm_evidence tests.test_round4_llm_policy tests.test_round4_llm_host_risks tests.test_round4_llm_harness tests.test_gateways tests.test_api_diagnostics tests.test_research_resilience tests.test_research_smoke -v
/workspace/scratch/b69ed5d5758a/qwen-test-env/bin/python experiment_round4_llm.py --output-dir /workspace/scratch/b69ed5d5758a/round4-llm-measurements
```

Windows equivalents (provided interface; not run on Windows in this round):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_round4_llm_evidence tests.test_round4_llm_policy tests.test_round4_llm_host_risks tests.test_round4_llm_harness -v
.\.venv\Scripts\python.exe .\experiment_round4_llm.py --output-dir C:\qwen_round4_mock
```

Python 3.12.14; openai 1.68.2, httpx 0.28.1, pandas 2.2.3; numerical environment numpy 2.3.5, scipy 1.17.0, scikit-learn 1.8.0. The last three differ from requirements.platform.txt (1.26.4/1.12.0/1.5.2). These are protocol checks in the available environment, not qualification of the official pinned numerical stack. No dependency files changed. Runs were sequential with at most eight configured BLAS/OpenMP threads; there was no separate heavy model search.

## Live boundary and handoff

`live_status=not_executed_no_credentials`: standard OPENAI_API_KEY and OPENAI_BASE_URL were absent. OPENAI_MODEL is optional; its absence is not treated as an error. No credentials were acquired, printed or copied. One existing owner command is:

```powershell
.\.venv\Scripts\python.exe .\experiment_rescue_api.py --execute --output-dir C:\qwen_round4
```

It checks the existing route; it does not activate this helper. All live stages/arms must share six actual attempts, 30,000 estimated tokens and 600 seconds, with 8192 completion reserve before admission, no SDK retries and only the existing one v3 recovery retry. A paired policy live experiment needs the captain's reviewed hook and enough remaining shared budget; otherwise only the integration gate is feasible. Live availability, live hypothesis validity rate and equal-budget production quality improvement remain unmeasured.

See [context map](round4_llm_context_map.md) and [integration request](round4_llm_integration.md) for the pure interface, bounded fields/history, trusted source requirements, pre-guard retry hook, original deadline and fail-closed local continuation. Review the evidence projection independently from the unproven stop policy. Exact-repeat suppression is promising protocol evidence; blanket stop-after-two is rejected as a generally safe default. Production activation/flag, full-main quality evaluation, merge and release remain with the captain. No other team's PR is a dependency.

Changed files are limited to the two new `agent/round4_llm_*.py` helpers, `experiment_round4_llm.py`, four `tests/test_round4_llm*.py` files, and `docs/round4_llm*` plan/map/integration/report/results. No `research*`, CLI, shared ML, requirements or builder file changed.
