# Qwen reliability: measured offline evidence, live check pending

- Base/main checked: `d05572846369f3b1e9cfa33f4008dda028db4fe0`.
- Tested runtime implementation: `e7f231e315959e05040dec11f440fe76e5b28fa0`.
- Branch: `work/rescue-qwen-v2`; final delivery SHA is the PR head. Subsequent
  commits add evidence/tests/docs, not a different host policy.
- Recorded replay: **2026-10-22 19:15:51–19:15:59 UTC** /
  **22:15:51–22:15:59 MSK**; clean source tree at replay time.
- Live Qwen calls in this work: **0**. Standard environment and local repo
  `.env` had no usable API configuration. No keys were requested or copied.

## Decision

The standalone probe and public evidence fixes are ready for captain review.
Context preparation is implemented but inactive. Do not claim live Qwen
reliability, measured live usefulness or a full live Qwen submission path.
Open the PR as a draft until the captain resolves the old fixture migration and
reviews the optional host integration. No changes to main or merges are made.

## Current request map (before and after this PR)

| Location | Existing responsibility / observed behavior |
| --- | --- |
| `llm.make_client` | Standard env; qwen/qwen3.8-27b default; preferred official gateway; SDK retry disabled |
| `gateways` | New official host then Railway; custom endpoint isolated; typed temporary errors only; shared-quota/account errors terminal |
| `v3._context` and `SYSTEM_V3` | Actual INNER research instructions, baseline recipe, diagnostics and three-file contract; unchanged |
| `V3State.request_allowed` | Original deadline/finalization reserve, step cap, 64 KiB dialogue cap, 100k estimated-token guard; reserves before each request |
| `v3._request_completion` | Refresh budget; timeout <=120s; increment attempts; one recovery retry across the run; each retry rechecks guards |
| Response handling | Increment responses before optional usage/envelope validation; missing usage stays unknown; malformed paid reply is not a transport retry |
| Tool loop | Length is terminal; invalid JSON becomes paired tool error without execution; successful Python alone is not verified INNER |
| Finalization | Close dialogue; freeze hashes; host outer admission; full refit of allowed recipe; validate and atomically publish CSV |

Found and fixed: the old smoke `BudgetClient` hid `base_url`, so it disabled
official route detection. It now forwards the endpoint through clones while
retaining a shared guard. Native contest-host retries/timeouts are unchanged.
`successful_experiments` was already LLM-only; its meaning is preserved. The
old generated-candidate receipt check did not prove Qwen origin when a local
candidate won later. The new summary matches verified origin plus both hashes.

The default model, gateway order, 10-second connect cap, <=120-second research
request cap, SDK retries=0 and one v3 recovery retry remain control policy.
No higher timeout/retry budget was adopted without evidence. `SYSTEM_PROMPT`
and `SYSTEM_V3` remain distinct and both unchanged.

## Validation

Captain integration review, 22 October 2026: the three old smoke fixtures below
were migrated without weakening the evidence gate. The complete targeted set,
including all 11 old smoke tests, passed **96 tests** in two invocations
(51 in 16.251 seconds and 45 in 0.989 seconds), with zero network/API calls.
This rerun used Python 3.12.14 and the repository's pinned numerical core:
NumPy 1.26.4, SciPy 1.12.0, pandas 2.2.3, scikit-learn 1.5.2, CatBoost 1.2.8
and LightGBM 4.6.0. AutoGluon was absent; this is not full platform qualification.
The isolated probe/public export review found no blocking issue. The receipt
summary trusts the native host's verified artifacts; it is not an independent
audit of context nonces, source bytes or full-fit input rows. Live Qwen behavior
and the inactive context helper's quality benefit remain unverified.

The original author validation below is retained as historical evidence.

The final targeted suite passed **85 tests in 21.685 seconds**, including the
supervisor, flat-ZIP imports and combined staged replay through the actual host.
The runtime source is the commit above; the later evidence-only commit adds
`tests/test_rescue_api_evidence.py` and these report/integration documents.

```powershell
python -m unittest tests.test_rescue_api_transport tests.test_rescue_api_context tests.test_rescue_api_host tests.test_rescue_api_smoke tests.test_rescue_api_probe tests.test_rescue_api_supervisor tests.test_rescue_api_evidence tests.test_gateways tests.test_api_diagnostics tests.test_llm tests.test_research_resilience -v
```

Coverage includes typed connect/read timeout; 500/502/503/504; regular 429 versus
quota 429; auth/permissions/schema; exhausted global retry; token/time/request
guards before retry; missing usage; empty/malformed/length output; invalid JSON;
custom-key isolation; hostile exception/header/body content absent from public
exports; baseline recovery and frozen candidate full refit. A hung worker and
its child are terminated by the parent deadline. Both new agent helpers and
the diagnostic helper import from the unchanged flat ZIP builder.

Host tests use real subprocess execution/scoring/finalization with mock API and
a cheap fixed startup baseline. This verifies protocol plumbing, not production
baseline search quality or provider availability. No heavy ML matrix was run.

Environment: Python 3.12.14; OpenAI 1.68.2, httpx 0.28.1, pydantic 2.10.6,
python-dotenv 1.0.1, psutil 6.1.1. Numerical runtime: NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, scikit-learn 1.8.0. Installing the pinned NumPy 1.26.4 wheel led to
SIGBUS during import in this execution host, before tests. The local override
was removed and tests rerun with the working runtime. **Exact official pinned
numerical-environment validation remains pending.** Requirements were not changed.

### Original old-suite migration failures — resolved in captain review

`python -m unittest tests.test_research_smoke -v`: **8 pass, 3 fail**.

| Existing test | Why the new evidence gate refuses its fixture |
| --- | --- |
| `test_completed_baseline_after_outer_candidate_failure_is_not_full_candidate_path` | Counter-only INNER and baseline publication without verified frozen event/finalization receipts |
| `test_explicit_execute_shares_request_budget_and_records_both_tasks` | Same synthetic success assertion without recipe/receipt proof; exit is now 1 |
| `test_generated_candidate_requires_matching_outer_refit_receipts_and_published_bytes` | Candidate hashes/receipts have no matching verified `origin=llm` INNER event |

Those old test files were outside the original author's explicit file ownership.
New `test_rescue_api_smoke` tests prove both full Qwen provenance and correct
baseline finalization after a losing Qwen trial. This is not an all-suite-green
claim for the original commit. The captain subsequently added verified LLM
INNER events and winning baseline/candidate receipts to these fixtures; all
11 old smoke tests now pass, including assertions that baseline publication
does not imply a Qwen full refit.

## Recorded staged mock probe

Real SDK with `httpx.MockTransport`, **zero network requests**, fixed test code
instead of a model proposal. Limits: six calls, 30,000 estimated tokens,
90 seconds. All rows below are simulated provider replies, not live Qwen.
Safe route label `alternate` means the configured route was preserved; it does
not imply that host was contacted. No errors/status/cause occurred in this replay.

| Attempt | Stage | UTC start (MSK = UTC+3) | Duration, s | Request bytes | Messages | Output cap | Token reserve | Result / usage |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | text | 19:15:51.234878 | 0.008955 | 100 | 1 | 64 | 98 | mock response / unknown |
| 2 | tool_schema | 19:15:51.245883 | 0.004603 | 567 | 1 | 256 | 445 | exact safe tool / unknown |
| 3 | research_shape | 19:15:51.284706 | 0.002665 | 3169 | 2 | 4096 | 5153 | structure valid, not executed / unknown |
| 4 | v3_smoke | 19:15:52.627029 | 0.003682 | 6530 | 4 | 8192 | 10369 | candidate tool / unknown |
| 5 | v3_smoke | 19:15:54.014311 | 0.004556 | 9611 | 6 | 8192 | 11396 | completion / unknown |

Total reserved estimate: **27,461**, never refunded. Five responses lacked usage;
these are not zero-token replies. Currency charge is not inferred from usage;
the replay did not contact the service. Local report:
`data/experiments/rescue_api_offline_evidence/probe_report.json` with source SHA,
UTC/MSK, full attempt metadata and strict public smoke summary. Native v3
`run.json`, receipts and generated files are under the same local evidence tree,
not committed/exported. Probe/smoke preflight reports are under
`data/experiments/rescue_api_preflight` and `data/experiments/rescue_smoke_preflight`.

The actual v3 stage made two mock API calls, executed one candidate and verified
one INNER trial, with no host-local/history trials. INNER RMSE:
`3.352800008858181e-15`. Host outer admission selected the fixture candidate;
both recipe hashes matched through full refit and the fresh validated CSV.
Final status `completed/model_finished`; elapsed 7.97 seconds.
This confirms the mocked route through all eight coverage milestones. It is
**not** live Qwen evidence or a competition-quality comparison.

Reproduce and retain a fresh, explicitly mock report (including the exact
current HEAD SHA and harness hash) with no network access:

```powershell
python -m tests.test_rescue_api_evidence --output-dir data/experiments/rescue_mock
```

The new replay's bytes, timings and estimates can differ with output path and
host-budget timing; do not substitute them for the historical table above.

## Context policy and usefulness

Same deterministic paired fixture: 150 repeated successful stdout lines shrink
from **12,748 to 3,980 bytes (68.78%)**, with all unique text, system constraints,
tool pairs, recipe, INNER metric and errors retained. The initial first-trial
hint instead adds **752 bytes** (2,170 to 2,922). Neither observation proves
faster experiments or fewer requests from live Qwen. The helper is inactive;
live old/new comparison remains pending after captain integration. See
`rescue_api_context.md` for exact preserved data and tests.

Inherited diagnostics from the assignment remain historical, not rerun here:
a tiny OK is not a measured experiment; prior 504/connect failures do not prove
their cause; the 22 October smoke had three attempts/one reply/two 504 errors
without a verified trial, and its ordering is not inferred. Platform score
improvement cannot identify Qwen contribution. No old report was modified.

## Next useful check

From the existing configured Windows checkout, once captain review is complete:

```powershell
.\.venv\Scripts\python.exe .\experiment_rescue_api.py --execute --output-dir C:\qwen_probe
```

This is bounded by six requests, 30k estimated tokens and ten minutes total;
the first failed prerequisite stops progression. It uses the ordinary local
configuration, not a command-line secret. Do not run another paid series after
quota failure. Exact integration/rollback and counter semantics are in
`rescue_api_integration.md`. Publish only the safe report, never native logs.
