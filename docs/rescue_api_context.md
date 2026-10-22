# Opt-in INNER context preparation

Base: `d05572846369f3b1e9cfa33f4008dda028db4fe0`.
The final implementation SHA and validation timestamp are recorded in
`rescue_api_report.md` / the PR. This helper is **inactive** in the competition
route until the captain connects it. Neither `SYSTEM_V3` nor the legacy
`SYSTEM_PROMPT` is changed by this patch.

`agent.llm_context.compact_research_messages` is a pure function with stdlib-only
dependencies. It accepts the existing, already-redacted v3 dialogue and the
integer number of remaining API attempts. It does not accept or inspect a full
host report, open files, obtain answers, query the API, change a deadline, or
execute model code. It returns `PreparedContext(messages, summary)`.

The transformation is deliberately conservative:

- All messages remain in order; every system message, task description, selected
  recipe, tool call and matching tool result remains present. Missing or duplicate
  call IDs are rejected before a prepared list is returned.
- Only consecutive exact repeated **stdout lines** from a successful process
  with a host-verified INNER metric are compressed. One complete occurrence and
  its repetition count remain. The replacement must save bytes. Every unique
  line, all stderr, all error records, candidate-execution failures, complete
  selection diagnostics, hashes and recipes remain intact. Large unique output
  remains large and can still hit the original context guard.
- Before the first verified INNER trial, at most one transient instruction asks
  for a small candidate and all three fresh files in one tool call. It keeps
  `fit_predict(X_fit, y_fit, X_predict, context)`, allowed-input rules, host
  measurement and the finalization reserve. With one or two attempts remaining
  it discourages additional broad analysis. This is an unmeasured hypothesis
  about model behavior, not evidence that Qwen will comply.
- Explicit non-INNER `selection.scope` is rejected. Unknown or malformed tool
  content remains unchanged and does not count as a verified trial. This helper
  is not an outer-data detector, a security sandbox or a public-export sanitizer:
  arbitrary free text cannot establish its own provenance. Supply only the
  pre-freeze v3 dialogue built from approved inputs; never feed a final report
  or outer/refit results into it. Redaction remains the caller's responsibility.

## Integration owned by the captain

In `research_v3._request_completion`, inside `while True`, prepare a fresh
temporary copy of accumulated `messages` **before** appending the existing
`Current host budget` snapshot and before `state.request_allowed`. Leave the
accumulated history unchanged. For an optional policy flag, the enabled branch
has this shape:

```python
prepared = compact_research_messages(
    state.clean(messages),
    remaining_requests=max(0, state.report["max_steps"] - state.report["api"]["attempts"]),
)
request_messages = [*prepared.messages, {"role": "user", "content":
    "Current host budget: " + json.dumps(state.budget_context(), ensure_ascii=False, allow_nan=False)}]
blocked = state.request_allowed(request_messages)
```

The unchanged request code must use `state.clean(request_messages)` and the
unchanged `TOOL`, `parallel_tool_calls=False`, model and 8192 output reserve.
Do not add `tool_choice`, streaming or extra request fields. Every retry repeats
this construction and the original guards; there is no second API coordinator.
The actual payload, including the transient hint and fresh budget, is what the
existing byte/token guard reserves. `summary.prepared_bytes` excludes that later
budget snapshot and is not a replacement for the guard or provider usage.

The summary contains only `policy`, `original_bytes`, `prepared_bytes`,
`stdout_runs_compacted`, `first_trial_hint_added`, `verified_inner_seen`.
These fields are suitable for aggregate request comparisons. The returned
messages themselves must never be copied into a public summary.

On helper validation failure, stop the paid dialogue through the existing
failure/local-continuation path with the original remaining budget; do not
silently send a dialogue rejected for a selection-scope or tool-pair violation.
No candidate admission, finalization, local continuation or ML-budget policy is
implemented here. Rollback is to disable the opt-in preparation branch and use
the original `messages` construction. The default remains the original route.

## Offline evidence and limits

Command: `python -m unittest tests.test_rescue_api_context -v`.
Thirteen tests passed under Python 3.12, including import from the normal flat
ZIP with no source `agent/` package or `PYTHONPATH`. Tests exercise actual
`SYSTEM_V3` preservation, all three filenames and function contract, recipes,
metrics, diagnostics, warning/error preservation, Unicode, malformed contents,
scope rejection, complete tool transactions, purity and no duplicate hints.

Same deterministic fixture and guard accounting convention (compact JSON,
UTF-8; no budget snapshot yet):

| Fixture | Original bytes | Prepared bytes | Interpretation |
| --- | ---: | ---: | --- |
| Verified INNER trial, 150 identical diagnostic lines plus one unique line | 12,748 | 3,980 | 8,768 fewer bytes (68.78%); all unique text, errors, recipe and metric retained |
| Initial v3 dialogue, two attempts remain | 2,170 | 2,922 | 752 extra bytes for the optional first-trial instruction |

This proves only payload reduction on a repetitive synthetic fixture and the
listed structural invariants. It does **not** prove fewer paid calls, faster
first INNER success, better scores, or compatibility of a live Qwen response.
The paired live comparison remains pending after host integration; keep the
same gateway, original deadlines, retry limit, model, schema and output reserve
and vary only this opt-in context preparation. Do not attribute locally trained
results or successful HTTP replies to Qwen experiments.
