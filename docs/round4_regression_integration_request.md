# Round 4 regression: integration decision

**Decision: retain the current rescue-v1 incumbent. No host change requested.**

Base: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`.
Tested experimental source: `be060588e820eec54c2cbacc96f998d4bcbb75cf`.

The frequency-only finalist improved RMSE on two development transitions and
reduced aggregate SSE. It advanced conditionally to one compatibility check:
the exact cold-object gate remained unknown because default discovery supplied
no schema on periods2/3/4. This was not native admission or a fully satisfied
development gate. The single post-freeze known-Year5 pair rejected it: incumbent RMSE
**39.471458**, finalist **47.640123** (about 20.69% worse). The known slice is
development evidence, not a hidden TEST estimate. No retuning or second
compatibility attempt followed this result.

The candidate is therefore an experimental adapter with a recorded negative
result. Do not replace the factory in `agent/research_extensions.py`, request a
new automatic CLI route, or spend a native900s A/B on this rejected variant.
Successful full refit and package tests establish executability; they do not
override the measured quality regression. A future proposal needs a new
predeclared hypothesis and evidence, rather than another weight adjustment on
the revealed compatibility slice.

## Existing behavior verified unchanged

At the tested SHA, the diff from the base consists of nine added files only.
All previously tracked files retain their base blobs, including:

- `agent/regression_base.py`, `regression_history.py`,
  `regression_candidate.py`, `regression_discovery.py`, `target_transform.py`;
- `agent/history*.py`, shared host/runner, CLI, models, features, metrics and splits;
- dependency lists, build code, classification and API modules.

The existing host still imports `candidate_source` and `make_regression_recipe`
from `agent.regression_candidate`; it constructs `rescue_regression_v1`.
No shared runtime or CLI file references `round4_regression`. The flat archive
contains the new modules as importable files, which does not activate them.
Existing CLI automatic rescue rollout and `--no-rescue-regression` control
remain the accepted incumbent behavior.

## Experimental interface, for reproducibility only

`agent.round4_regression_candidate.make_round4_recipe(...)` and
`candidate_source()` expose the unchanged `fit_predict` contract under explicit
family `round4_regression_v1`. The rejected finalist uses the incumbent CatBoost
base and fit-only category frequencies, with missing indicators disabled and
legacy history calibration. No new discovery budget or host hook is required
or requested.

The complete algorithm, configuration variants and measured failures are kept
with `round4_regression_recipes.json` and `round4_regression_results.md`.
The final full-refit retry completed 53,308 fit rows and 22,449 predictions in
11.582 seconds with two physical fits; this is an observed contract run, not a
minimum host-budget recommendation. Resource qualification remains limited by
available telemetry.

Local validation: 31 new focused tests and 78 incumbent regression tests passed.
A frequency-only synthetic smoke from the flattened package, outside the source
repository, passed three serial invocations/six physical fits with a 50-tree
toy cap. It verified fresh full refit, sidecars, finite predictions and prediction
order/batch independence. It is not a quality comparison.

**Merge and release remain the captain's decision. No production promotion is
recommended for this round's candidate.**
