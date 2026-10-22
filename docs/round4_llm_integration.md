# Round 4: optional INNER evidence and next-action policy

Base: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`.
Policy: `round4_inner_action_v1`. Implementation/evidence HEAD is recorded in
`round4_llm_report.md` and the PR. The production host does not import or activate
this policy. Host-owner review and an explicit optional flag are required before
activation. The round-4 harness activates its adapter only for local mock work;
that is not an activated contest route or a live-quality measurement.

## Pure interface

```python
from agent.round4_llm_policy import prepare_round4_messages

prepared = prepare_round4_messages(
    messages,                       # already-cleaned pre-freeze v3 dialogue
    remaining_requests=remaining,   # integer 0..128, actual remaining API attempts
    inner_snapshot=snapshot,        # allowlisted trusted INNER aggregate projection
    experiment_history=history,     # at most twelve host-owned observations
)
```

The result is `PreparedPolicy(messages, policy_version, action, stop_reason,
summary)`. `action` is `create_candidate`, `inspect_one_field` or `stop_local`.
No action is executed by this function. All original messages, system rules,
assistant tool arguments and tool results remain bytewise unchanged in the
returned prefix. Inputs are not mutated. The appended evidence/instruction is
temporary; it must never be accumulated into subsequent conversation history.

The helper does not accept `state`, a full report, filesystem paths, outer data,
target arrays, a client or a clock. It reads no file/environment and trains no
model. These are deliberately limited inputs: a scope string or hash alone does
not authenticate provenance. The host must construct the projection from its
approved pre-freeze research state. The schema cannot detect an outer score or
secret deliberately mislabeled as an INNER number or a literal class label.

## Input fields and sources

`validate_inner_snapshot` defines the version-1 snapshot; the complete current
context/manifest map and exact field caps are in `round4_llm_context_map.md`.

| Field | Trusted source and scope | Missing behavior |
| --- | --- | --- |
| Task type, metric/direction, prediction kind, class/probability order | Validated research TaskSpec; schema, not row answers | Required; reject null/incomplete contracts |
| Allowed inputs, feature roles, fit/validation counts, split | Approved manifest and INNER split; no row IDs or index arrays | One narrow manifest inspection at most |
| Baseline recipe and its identities | Already frozen research baseline | Null means unknown; no extra fit to populate it |
| Current metric and error aggregates | Completed host INNER measurement/feedback only | Null; never copy final report or substitute outer |
| Remaining time, reserve, measured fit time, resources | Current host budget; measured timing, not claims by the model | Null stays unknown; original guards remain authoritative |
| Installed tools | Host availability check against permitted package names | One narrow availability inspection at most |
| Optional feature profile | Explicit feature-only INNER-fit rows | Null or bounded partial profile; never required for a candidate |

The mandatory task contract is not guessed. Optional absent baseline,
diagnostics, profile and measured fit cost do not trigger extra inspection.
The profile helper accepts at most 512 rows, 32 columns/16,384 cells and a small
time allowance (default 0.05 seconds, maximum 0.25), charged to the same host
deadline. This is a cooperative loop bound, not a hard process sandbox. It
outputs counts, never individual values, targets, CSVs or unique-value lists.

History rows are explicit observations, not arbitrary ledger/report objects:

```python
{
    "scope": "inner_selection",
    "outcome": "verified_inner",
    "identity": {"code_hash": "...", "recipe_hash": "...", "recipe": {...}},
    "metric": {"name": "RMSE", "value": 0.7, "higher_is_better": False},
    "elapsed_seconds": 0.2,
    "accepted": False,
}
```

`outcome` must be one of `analysis_only`, `no_candidate`, `candidate_invalid`,
`execution_failed`, `narrow_inspection`, `verified_inner`, `truncated_response`,
`transport_error`. A `verified_inner` observation requires a metric matching the
task and a frozen candidate identity. Other outcomes cannot carry a verified
metric or `accepted=True`. Optional elapsed time is finite/nonnegative or null.
A `narrow_inspection` may include `inspected_field` naming the one requested
aggregate. Unknown free-text fields, raw exception strings and final-report
sections are rejected; original important errors remain in original tool results.

Identity is the actual candidate **source hash**, not the run_python wrapper's
code digest. The recipe hash is the host's normalized full recipe identity:
`sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':'),
ensure_ascii=False, allow_nan=False).encode('utf-8'))`. When full settings are
supplied, their hash is checked. If unavailable, omit `recipe`; keep the trusted
host hash. Never hash a family/settings summary and call it the complete recipe.
Model name similarity does not establish identity; different settings or source
hashes are distinct experiments. This helper does not alter host deduplication
or prevent execution of an unseen proposed duplicate.

`verified_inner` means a host-scored valid prediction contract. It does not
prove a physical training call: a constant or phase-sensitive function may still
produce valid predictions. Physical-fit counts require separate instrumentation
and are unknown when unavailable. A successful Python process alone is recorded
as a nontrial when no candidate was verified, never as a fit or improvement.

## Fixed next-action policy

The thresholds were selected before the paired mock run and are not tuned from
its confirmation outcomes:

1. Stop if no API attempts remain or reported research time is <=2 seconds
   (including remaining time minus finalization reserve).
2. Stop after any repeated exact **verified** code/recipe identity, or after
   two consecutive `analysis_only`/`no_candidate`/`candidate_invalid`/
   `execution_failed` observations.
3. If an optional essential aggregate is absent, request one narrow inspection.
   If an inspection has already happened and essentials are still absent, stop.
4. Otherwise request one concrete small candidate, hypothesis and recipe in a
   single tool call, using supplied fit arguments, allowed installed tools and
   measured costs. Do not read original training data inside the recipe, memorize
   row positions, branch on phase, or write the host's final CSV.

There is an explicit tradeoff: a useful proposal that would arrive after two
nontrials is not reached under this policy. The `late_useful` mock case records
that negative outcome. Do not interpret fewer requests as universal usefulness
or silently relax the threshold after seeing that case. The policy remains an
optional experiment until paired live/quality evidence justifies adoption.

## Sizes, errors and public observations

- Snapshot <=16 KiB; at most 64 features and 64 literal classes; full recipe
  <=8 KiB with bounded nested structure. Oversize is rejected, not truncated
  into an incomplete prediction contract.
- History <=12 rows and <=12 KiB; the complete appended evidence/instruction
  <=32 KiB. Existing dialogue is never truncated to make it fit.
- Missing/mismatched tool pairs, unsupported roles, explicit non-INNER scope,
  unknown history fields, invalid metrics/identities and exceeded caps raise
  `ValueError` with fixed safe text.
- Summary contains version/action/stop reason, byte counts, counts of verified
  observations/unique identities/duplicates, unproductive streak, and bounded
  names of missing/selected fields. It contains no prompts, tool arguments,
  recipes, feature values or classes. `prepared_bytes` is serialized dialogue
  bytes before the fresh budget message; it is not usage or a token guard.

The evidence can increase payload size. Source messages already contain some
facts, and the extra projection makes those facts explicit; there is no claim
that this policy compresses every prompt or lowers every request's token reserve.

## Host-owner integration request

The precise production hook is inside the existing
`research_v3._request_completion`, at the top of its `while True`, **before**
appending `Current host budget` and before request/time/token guards. The owner
constructs a projection at this hook from live pre-freeze state, not by reading
`run.json`. The ordinary `state.clean` redaction still applies.

```python
try:
    prepared = prepare_round4_messages(
        state.clean(messages),
        remaining_requests=max(0, state.report["max_steps"] - state.report["api"]["attempts"]),
        inner_snapshot=approved_inner_snapshot,
        experiment_history=approved_inner_history,
    )
except ValueError:
    state.event("llm_policy_rejected", policy="round4_inner_action_v1")
    return None, "api_error"  # existing local-continuation/finalization route

if prepared.action == "stop_local":
    state.event("llm_policy_stop", details=prepared.summary)
    reason = ("time_limit" if prepared.stop_reason == "research_time_exhausted" else
              "step_limit" if prepared.stop_reason == "attempts_exhausted" else "repeated_failures")
    return None, reason

request_messages = [*prepared.messages, {"role": "user", "content":
    "Current host budget: " + json.dumps(state.budget_context(), ensure_ascii=False, allow_nan=False)}]
blocked = state.request_allowed(request_messages)
# Continue the existing timeout, attempt accounting and SDK request unchanged.
```

This is an integration proposal, not code already added to `research_v3.py`.
The host must rebuild the snapshot and remaining-attempt count on every retry.
Its original guard must reserve the actual prepared payload including the fresh
budget message; a client wrapper cannot substitute messages after this check.
Use the original model, TOOL, `parallel_tool_calls=False`, 8192 output reserve,
10-second connect cap, <=120-second request ceiling and one v3 recovery retry.
Do not add tool_choice, streaming, a second deadline or a new local-search budget.

Invalid context is never resent unchanged as a fallback. The original host
loop receives the returned stop reason, retains input/checkpoint integrity
checks, and continues/finalizes with the original remaining deadline. Existing
nonadaptive handling of `time_limit` finalizes; adaptive handling may continue
locally if its original budget admits work. Neither path restarts the budget.

Host review tests must confirm actual prepared messages are guard-charged on
every attempt, pair/scope rejection sends zero paid requests, lost candidates
do not receive forced refit, and local winners retain local provenance. The
mock harness uses a thin temporary adapter before the original request function
for comparison; it neither copies the host nor changes the contest default.

Rollback disables the optional adapter/flag and restores the original
construction from `messages`. Keep the existing regression rollout, current
baseline, classification path and offline behavior. Dependencies: the evidence
and policy modules in this PR; production activation belongs to the captain's
separate host integration. No other team's working branch is required.

## Reproduction and live limit

Local structural checks:

```powershell
python -m unittest tests.test_round4_llm_evidence tests.test_round4_llm_policy -v
```

The report records the full paired harness command actually executed and its
source/input/split hashes. Synthetic canned replies are mock protocol evidence,
not replay of today's service or evidence of better real Qwen hypotheses.

If a standard local key is present, the already-existing authorized integration
command remains:

```powershell
.\.venv\Scripts\python.exe .\experiment_rescue_api.py --execute --output-dir C:\qwen_round4
```

All stages and variants share six real attempts, 30,000 estimated tokens and
600 seconds. There is no authorization for two independent six-request arms.
Since this policy is not in the production request hook, this command currently
checks the existing main route. A paired live policy comparison awaits the
captain's reviewed hook and enough shared budget; otherwise report live quality
effect as unmeasured. No local key means `not_executed_no_credentials` and does
not block delivery of the offline work.
