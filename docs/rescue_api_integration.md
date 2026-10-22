# Qwen rescue: integration owned by the captain

Base: `d05572846369f3b1e9cfa33f4008dda028db4fe0`.
Runtime implementation: `e7f231e315959e05040dec11f440fe76e5b28fa0`.
See `rescue_api_report.md` and the PR head for subsequent evidence/documentation commits.

## Active changes and inactive helper

| Change | Actual route | Activation |
| --- | --- | --- |
| `llm.public_error_message` | Public probe/smoke summaries | Active in those scripts only; native host diagnostics unchanged |
| Smoke wrapper `base_url` | `experiment_research_smoke -> run_agent -> v3._request_completion` | Active: restores the existing official failover policy |
| Strict provenance/receipt export | Smoke and probe reports after host finalization | Active; report version 2; incomplete old fixtures no longer pass |
| `api_probe.ProbeClient` / `ProbeBudget` | Standalone opt-in probe | Active only in probe; not installed in contest host |
| `llm_context.compact_research_messages` | Proposed v3 context preparation | **Inactive**, including in the probe's v3 stage |

`SYSTEM_PROMPT`, `SYSTEM_V3`, gateway order, SDK retries=0, the 10-second
connect timeout, 120-second research request ceiling and one global v3 recovery
retry are unchanged. The actual current v3 output reservation is **8192** tokens.
No streaming, `tool_choice`, nonstandard request options or dependency changes.

## Thin host integration (optional, default off)

The captain owns `agent/research_v3.py`. Do not copy the host into this helper.
In `_request_completion`, inside its existing `while True`, replace only the
construction of the temporary dialogue immediately before the fresh budget
snapshot. Supply `state.clean(messages)` and
`max(0, state.report['max_steps'] - state.report['api']['attempts'])` to
`compact_research_messages`. Use `prepared.messages` when the optional policy
is enabled. Keep original accumulated `messages` unchanged.

Append the existing `Current host budget` message, then execute the unchanged
`state.request_allowed(request_messages)`, `state.research_timeout(120)`,
attempt increment and SDK request. Every retry must repeat these guards using
the actual prepared payload. Never use `prepared.summary` as a token counter.
Its byte counts exclude the later budget snapshot and tool schema.

The helper returns `PreparedContext(messages, summary)`. The summary contains
only policy, before/after byte counts, repeated-line count, hint-added flag and
verified-INNER-seen flag. It reads no files, environment or full host reports.
The input is the existing pre-freeze INNER dialogue, never an outer/final report.
All three fresh artifact filenames and the exact `fit_predict` contract remain.
System constraints, tool/result pairs, recipes, unique output and error details
remain. Only exact repeated stdout after host-verified success is compressed.

If helper validation rejects a pair or non-INNER selection, stop the paid
dialogue via the existing local continuation/finalization path. Do not restart
the ML budget or send a rejected dialogue unaltered. The implementation of this
handling belongs to the captain, alongside the optional flag. Rollback disables
the optional branch and restores the original request-message construction.

Required integration checks: unchanged SYSTEM_V3; paired tools; no outer input;
same code/recipe before outer/refit; original guard charged each attempt;
custom endpoint stays isolated; same original deadline after API failure.
Compare old/new with the same task, model, route, retry policy and output reserve.
Do not simultaneously change timeout or increase retries. The first-trial hint
is a hypothesis; it adds bytes and has not demonstrated a live benefit.

## Probe orchestration and accounting

Run from the existing configured checkout on Windows:

```powershell
.\.venv\Scripts\python.exe .\experiment_rescue_api.py --execute --output-dir C:\qwen_probe
```

No copying keys into the command; the standard client loads local `.env` with
environment precedence. Omit `--execute` for a zero-network preflight. Unknown
quota is `null`, not zero/unlimited; no documented balance endpoint is assumed.
Missing settings block execution before an SDK client or worker is created.

The stages are text (64 output tokens), exact deterministic `run_python` tool
(256), research-shaped request (4096, actual SYSTEM_V3/schema/required fields),
then one synthetic regression through unmodified v3 (8192 per response). The
research-shaped response is structurally checked but **never executed**. Only
the exact local `print("rescue-api-tool-ok")` constant runs in the tool check;
arbitrary research code executes only through the existing v3 process controls.
There is no claim of an OS sandbox.

The three diagnostics share at most one recovery retry. v3 keeps its own one
global recovery retry; every attempt from both coordinators goes through the
same **6 requests / 30,000 estimated tokens / 600 seconds** ceiling. These are
upper bounds. Exhaustion and invalid paid output stop progression. Quota,
authorization and schema errors do not trigger route switches or a second
series. A retry can consume the space that would otherwise be used for smoke.

`ProbeBudget` reserves `ceil(full UTF-8 request JSON bytes / 3) + max_tokens`
before the underlying call, including tools and response reserve. No reservation
is refunded on failure or missing usage. The top-level attempt list is the
authoritative probe transport count. Native v3 counts a host invocation before
the wrapper; when the shared ceiling blocks it, native `api.attempts` can be
one higher than actual sent smoke requests. Keep both, **do not sum them**.
`usage_complete` concerns received responses only; failed calls can have unknown
billing. Currency cost remains unknown. `in_flight` after a killed worker means
dispatch was reserved but response outcome is unknown, not a confirmed success.

The v3 stage receives at most 300 seconds and only the remaining shared time;
it retains its finalization reserve and local continuation. A parent supervisor
also enforces the whole-worker deadline, samples memory and cleans tracked
descendants using the existing `ProcessTreeMonitor`. Bounded cleanup may take
up to roughly three additional seconds. A hard stop cannot claim completed
finalization; any previously published checkpoint remains local.

## Evidence and migration

Public `probe_report.json` includes source SHA/fingerprint, UTC/MSK times,
attempt index, stage, safe route name, duration, request byte/token estimate,
limits, typed error/status/cause and known usage. It excludes request IDs,
prompts, bodies, headers, candidate source, CSV and personal paths. Native logs
and `.agent_runs` remain local and are not safe public exports.

`inner_trial_verified` retains the native LLM-only experiment counter. Frozen
and published provenance additionally matches host-verified INNER event hashes
and outer/full-refit receipts. A local winner after a Qwen trial receives local
provenance. Baseline publication after a losing Qwen candidate can pass INNER
integration without refitting that loser. Full Qwen publication is a separate
claim requiring the same admitted Qwen recipe and matching fresh CSV digest.

The captain must update three pre-existing `tests/test_research_smoke.py`
fixtures outside this branch's file ownership: add verified `origin=llm` INNER
event identities and genuine winning-recipe outer/refit receipts, or expect
incomplete evidence. Names and observed failures are listed in the report.
Never weaken the new checks just to accept counters or hash-shaped strings.

No platform score, live tool compatibility, live INNER trial or live payload
efficiency is established by this PR. One bounded probe after review is the
next useful check; do not repeat paid series automatically after failure.
