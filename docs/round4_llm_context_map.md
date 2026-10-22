# Round 4: current context and bounded INNER evidence

Base: `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb`.
Implementation/head provenance and measured comparisons are in the round-4
report/PR. This map describes that fixed base, not the older rescue branch.
The new helpers are inactive in production until the host owner connects them.

## What the current host already supplies

| Existing source | Information already available to Qwen | Remaining friction |
| --- | --- | --- |
| `SYSTEM_V3` | Three fresh files in one tool call; exact `fit_predict`; raw-string X; training only on passed arguments; labels/probabilities; host scoring/refit/publication; no outer feedback | Analysis-only or contract-violating output despite these instructions is a compliance failure, not absent instructions |
| `ResearchWorkspace._task_text` | Task type, primary metric, exact class order, positive class, allowed CSV filenames, source-description preview and location | Direction/prediction contract and feature roles are not all repeated in the initial task text |
| `research_v3._context` | Frozen baseline recipe, current host INNER metric and split, baseline INNER diagnostics, explicit history trial if requested | No compact structured feature-count/profile or bounded per-attempt identity ledger |
| `research_validation.json` | Feature/numeric/categorical roles; classes/positive class/output probability mapping; allowed inputs; INNER row positions; split; three-file contract | Qwen may spend a tool call reading declarations that the host already has; raw row positions should remain in the manifest, not be copied into the new evidence |
| `inner_feedback` | Host-verified INNER aggregate regression error quantiles/bias, or bounded class performance/confusions | Available only after an actual host-scored trial; an arbitrary printed score cannot substitute |
| Fresh `state.budget_context()` per request | Remaining attempts/time, measured trial costs, finalization estimates and reserve | These are already refreshed before original guards; the helper cannot replace or discount them |

The source task can contain important domain conditions. Existing system/task
messages remain untouched; the evidence is not a substitute for the full
permitted task description. The host adapter should construct a small projection
from these pre-freeze objects. It must not read final `run.json`, whose later
fields include outer results and private paths.

## Proposed schema and exact provenance

`agent.round4_llm_evidence.validate_inner_snapshot(snapshot)` validates/copies
one explicit projection. Required header: version=1, scope=`inner_selection`.
Required task contract: registered metric and correct direction/type/prediction
kind; complete nonempty literal classes for classification; complete probability
column order for probability output. A single binary positive-probability column
is allowed only when it matches the declared positive class. Unknown mandatory
task facts fail closed. No truncated class list is accepted.

| Field | Trusted producer/source | Scope | Bound / purpose / missing behavior |
| --- | --- | --- | --- |
| `task` | `gate.spec.metric`, `.classes`, `.positive_class`, `.probability_columns` | Declared task contract | Registered metric; max 64 classes, 200 UTF-8 bytes per literal; preserves exact order; required facts cannot be null |
| `allowed_inputs` | `gate.manifest['allowed_inputs']` | Approved research view | Only the seven fixed host relative filenames; missing=null; excludes personal/absolute paths |
| `features` | `gate.spec.features/numeric/categorical` | Raw feature schema, excluding ID and target | Max 64 unique names, 200 UTF-8 bytes each; roles numeric/categorical only; missing=null |
| `shape` | Lengths of `gate.fit_rows`, `gate.held_rows`, `gate.features` | INNER fit/validation counts | Finite nonnegative integer counts; individual positions/IDs are not accepted; absent counts=null |
| `split` | `gate.inner_split` projection | INNER split only | Strategy and known seed only; actual base uses random/time/grouped; missing strategy/seed=null, never guessed |
| `baseline` | Already frozen `state.baseline` code/recipe identity and settings | Prepared recipe selected using allowed research view | Both SHA-256 identities, canonical settings <=8 KiB; missing=null; never read a raw final report |
| `current_metric` | `state.best_inner` before freeze | Host-verified INNER selection | Registered metric/direction, finite value; must match task metric; missing=null |
| `diagnostics` | Previously verified `inner_feedback` result | INNER aggregate errors | Exact aggregate field allowlist; max 12 class rows/8 confusions; no row errors/IDs/answers; missing/unavailable=null |
| `timing` | Current original budget/finalization plan and known measured INNER cost | Existing run budget | Numeric remaining/reserve/max-experiment/last-fit seconds, known memory limit and <=8 CPU threads; unknowns=null; memory limit is not peak RSS |
| `installed_tools` | Host inspection of the actual configured environment | Local installed packages | Names numpy/pandas/scipy/sklearn/catboost/lightgbm/xgboost only; missing=null; no claim based solely on requirements |
| `profile` | Optional `profile_inner_fit` below | Only explicitly passed INNER-fit features | Bounded counts, never raw values; missing=null; no training needed |

Total normalized snapshot limit: **16 KiB UTF-8**. Unknown fields are rejected.
Optional sections default to null. Optional count/timing members also normalize
to null. The helper receives no `state`, report, path, environment or filesystem
handle. A declared scope string is not evidence of provenance: trusted host
construction and the pre-freeze call location are essential. A secret disguised
as a legitimate class/feature name cannot be discovered by schema validation;
the host must preserve its existing redaction/provenance boundary. This is a
model-context API, not a general public-log sanitizer.

`validate_recipe_identity` uses exactly the host's canonical serialization:
sorted keys, compact separators, `ensure_ascii=False`, finite JSON, SHA-256.
Code hashes come from host-frozen bytes; this helper never reads candidate code.
Recipe settings are bounded to depth 8, 512 nodes, 64-item sequences, 200-byte
string values and 8 KiB total. Obvious private path/source/secret/outer-data
fields are rejected. A supplied full recipe must match its hash; a hash-only
past identity stays opaque, never recomputed from an abbreviated summary.
Different settings retain different identities, even within one model family.
If a legal recipe exceeds this context bound, retain original messages and
report the evidence preparation error to the host's local-continuation path;
do not silently trim the recipe into a new identity.

## Optional feature-only profile

The producer may call:

```python
profile = profile_inner_fit(
    gate.research_train.iloc[gate.fit_rows].loc[:, list(gate.features)],
    [{"name": name, "role": "numeric" if name in gate.numeric else "categorical"}
     for name in gate.features],
)
```

Only that explicit feature-only frame is accepted: extra ID/target columns or
schema drift are rejected. The producer owns the row-scope guarantee; the
helper cannot infer whether a mislabeled frame actually contains holdout rows.
It outputs per-column missing, distinct and numeric-finite **counts**; never
examples, per-row values, target distributions, IDs or errors.

Defaults: at most 512 rows, 32 columns and 16,384 cells, cooperative 0.05 seconds
(configurable only up to 0.25 seconds). Time is checked per cell; strings longer
than 4096 characters and unsupported scalar types cause a column to be omitted.
An incomplete column is omitted whole; it never reports partial counts against
a full sample size. `limited` and omitted-column count are explicit. If no
column completes, sample_rows=0. No fits, network requests or package installs
occur. The measured elapsed time is part of the original host budget. The
cooperative limit is not an OS timer and excludes upstream frame slicing;
the host must bound/charge that construction too and skip optional profiling
when its research reserve is already exhausted.

## Evidence checks

`python -m unittest tests.test_round4_llm_evidence -v` passed nine tests in the
working Python 3.12 environment. Tests verify immutable inputs, exact probability
labels/order, registered metric direction, absent versus zero aggregates,
scope/unknown-field rejection, canonical identity/tamper rejection, exclusion
of raw profile values/target columns, and cooperative row/cell/time limits.
An actual `ResearchWorkspace` intervention changes only the fixed outer-holdout
targets and rebuilds the workspace with the same frozen split: INNER row lists,
answers and the projected evidence/profile remain identical (timing excluded).
No model fitting or API calls: metrics and physical fits for these boundary
checks are **not_applicable**. Numerical environment qualification is reported
separately by the root report; these checks do not establish live Qwen quality.
