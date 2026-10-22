# Round 4 regression: retain the integrated incumbent

**Decision: reject the frequency-feature finalist for production.** It improved
aggregate development SSE by 1.70% over periods 2–4, but the single frozen
known-period-5 check worsened RMSE from **39.471458 to 47.640123 (+20.6951%)**.
The loss is concentrated in objects without past observations and new
categories. No change was made after seeing period 5 and it was not rerun.
The recommended candidate remains the existing `rescue_regression_v1`.

PR #44 supplies a separately versioned experimental adapter, bounded runner,
tests and negative evidence. It is not a request to enable the new family.
No new native A/B, platform submission or main merge was performed.

## Identity and comparison

- Base/current main checked at start and after measurements:
  `e62fea1bb30f1b5c2fd1663ad6421bd00280f7cb` (tree
  `59024a4f470939f01b4ba66a2dd6b27bfbd5718f`). The previously accepted rescue
  integration is already present; historical `integration_pending` notes do not
  describe this base.
- Preregistered plan commit: `7281618543af1726bfe59170d847c33aef1cd17f`;
  document SHA256 `aac67f185651d73be8590ea8adc67be57ff4e1d2d1a86f777b48a08aba116443`.
- All public model measurements used clean code
  `be060588e820eec54c2cbacc96f998d4bcbb75cf`, tree
  `1c507c51c277a9d3dcf97e2c48a614a39aa5679a`. Subsequent evidence commits only
  add documentation/data-free result files.
- Serial CPU fits, 8 threads, seed 42, identical 180-second invocation allowance
  including host preparation and fresh refit. No API calls. Requested memory
  allowance 8 GiB; RSS, process CPU and cleanup were **not observable**.
- The control is the current fixed rescue algorithm retrained on identical
  supplied fits, not a replay of the entire native scheduler or an old weak seed.
  These are previously exposed public TRAIN slices, not a blind competition score.

`round4_regression_results.json` contains source/input/split/recipe/output hashes,
all runs, phase times and physical fit ledgers. The CSV is a compact run table.
`round4_regression_recipes.json` contains both standalone entry-point source
strings, all screen recipes, the recommended existing recipe and the rejected
finalist recipe. A saved history key is a result of host discovery for this
dataset, not a production hardcoding. Reproduction and integration notes are
separate adjacent documents.

## Frozen nesting and scope

For external period t, train on periods strictly earlier than t. Host discovery
uses only that fit with its latest period removed, with the current **8M/48**
bounds. Candidate early stopping/calibration independently holds the latest
period inside its supplied fit. No 12M/192 helper or discovery inside the worker
was used. The schema was frozen once and shared by both arms.

| External period | Supplied fit | Prediction rows | Host discovery rows | Result |
| --- | ---: | ---: | ---: | --- |
| 2 | 2,117 | 9,058 | 0 | Abstained: no rows |
| 3 | 11,175 | 10,879 | 2,117 | Abstained: single period |
| 4 | 22,054 | 12,946 | 11,175 | Limited: proposer budget |
| 5 | 35,000 | 18,308 | 22,054 | Structurally eligible schema |

Thus the four exact history groups are **not assessable** on development
periods 2–4. A `schema_unavailable` group covers all rows there; it is not a
claim that all objects are cold. The development cold-object gate remained
unknown, not passed. A novel-category proxy was inspected, but cannot replace
the history groups. Ffrequency advanced only to the preregistered compatibility
check, not to native admission. On period 5 the shared discovered key was
`floor_area, year_built`, with strict past-only, past-unique eligibility.

## Bounded screen and development result

All H1 base variations failed the first period-4 screen, so the conditional
base-plus-feature combination was skipped. Missing flags alone and combined
with frequencies failed. The LightGBM pair was a preregistered calibration
diagnostic, not a new-family selection contest. Ten unique configurations were
tested, below the eleven declared maximum. Only Ffrequency became a finalist.

| Period-4 arm | RMSE | Best / final trees | Physical fits | Host total, s |
| --- | ---: | --- | ---: | ---: |
| Incumbent | 46.824893 | 66 / 66 | 2 | 7.493 |
| B4: depth 4, lr .05, L2 10 | 48.502876 | 37 / 50 | 2 | 6.256 |
| B5: depth 5, lr .04, L2 10 | 47.940465 | 68 / 68 | 2 | 6.505 |
| B7: depth 7, lr .04, L2 10, accepted replay | 47.009759 | 105 / 105 | 2 | 7.619 |
| B6R: depth 6, lr .03, L2 20 | 47.081582 | 133 / 133 | 2 | 8.451 |
| Missing flags | 47.649827 | 42 / 50 | 2 | 7.805 |
| Category frequencies | 46.250038 | 90 / 90 | 2 | 8.517 |
| Both blocks | 46.836712 | 53 / 53 | 2 | 8.865 |
| Existing LightGBM | 45.605542 | 31 / 50 | 2 | 6.571 |
| LightGBM calibration aligned to refit count | 45.605542 | 31 / 50 | 3 | 6.993 |

The original B7 invocation completed two fits but its input-integrity check
failed; its score was never admitted. Its unchanged technical replay appears
above. All failed attempts remain in the accounting below.

| Period | Incumbent RMSE | Ffrequency RMSE | Relative RMSE change | SSE change |
| --- | ---: | ---: | ---: | ---: |
| 2 | 49.752709 | 47.268174 | −4.9938% | −2,183,447.29 |
| 3 | 55.794171 | 56.962942 | +2.0948% | +1,433,713.14 |
| 4 | 46.824893 | 46.250038 | −1.2277% | −692,670.24 |
| 5, known compatibility only | 39.471458 | 47.640123 | +20.6951% | +13,027,704.07 |

Across development periods 2–4, SSE fell from 84,672,789.95 to 83,230,385.55:
−1,442,404.40 (−1.7035%), with two wins out of three. Period-3 loss was mainly
in the lower half and p50–p90 target bands; the tail above fit p99 improved.
Novel-category SSE rose 9.7903% there, with only 833 rows; this proxy does not
establish the declared cold-group gate. Selected trees changed 152→308. This pattern
does not establish a robust improvement or justify clipping outliers.

For period 2 both arms used their single-period fallback of 400 trees and one
full fit. Periods 3/4 used two fits per arm. With no history schema, base, fixed
mean and calibrated outputs coincide on these development slices.

## What failed on the one known-period-5 check

| Policy | Incumbent RMSE | Ffrequency RMSE |
| --- | ---: | ---: |
| Base only | 44.609419 | 51.648012 |
| Fixed history mean 0.5 | 39.475939 | 47.563824 |
| Fit-only calibrated history | 39.471458 | 47.640123 |

Both arms independently selected the same history rule: last observation,
weight 0.75, shrinkage 0, from 19 fixed choices. Calibration had 12,946 rows,
9,540 eligible; outer prediction had 12,053 eligible and 563 ambiguous rows.
Internal calibration RMSE was 31.4013 vs 31.1978, while final base trees were
570 vs 495. A slightly better internal result did not transfer to the next
period. The loss already exists in the base and persists under fixed mean;
changing the history weight is not supported by this evidence.

The exact disjoint history groups below use calibrated predictions. Shares are
of each arm's own total SSE; counts sum to 18,308 in each arm.

| Group | Rows | Incumbent RMSE / MAE | Incumbent SSE / share | Ffrequency RMSE / MAE | Ffrequency SSE / share |
| --- | ---: | --- | --- | --- | --- |
| Eligible | 12,053 | 23.1715 / 12.0191 | 6,471,480 / 22.6880% | 23.0446 / 11.9560 | 6,400,765 / 15.4044% |
| Ambiguous past | 563 | 32.1697 / 18.0138 | 582,643 / 2.0427% | 32.0131 / 17.8958 | 576,984 / 1.3886% |
| Valid key, no past | 5,593 | 61.6170 / 41.6508 | 21,234,673 / 74.4455% | 78.3622 / 58.1103 | 34,344,572 / 82.6554% |
| Incomplete key / invalid period | 99 | 48.7204 / 35.2919 | 234,994 / 0.8239% | 48.1133 / 33.1283 | 229,174 / 0.5515% |

The 5,593 valid objects without past observations account for +13.110M SSE
(+61.738%, beyond the 10% cold-group limit), slightly more than the net loss
because other groups improved. Independently,
the 3,192 rows with any new category account for +12.941M SSE (99.34% of net
loss). These partitions overlap and must not be summed. New-category RMSE rose
63.7798→90.1229. A dynamically identified previously unseen State_Factor level
contains 3,191 of those rows; mean truth was 53.48, versus mean predictions
89.09 and 117.29. This observation is descriptive, not a permitted new
dataset-name/category switch.

Frequency features encode unseen categories as zero while known categories
have positive fit frequencies. Extrapolation under category distribution shift
is a plausible explanation, **not an identified causal mechanism**. No new fit
or changed recipe was used to test that explanation after seeing period 5.

Earlier period-4 incumbent diagnostics showed 113 rows above fit p99 (0.873%)
contributing 51.60% of SSE and substantial underprediction. Common categories
contributed 90.20%, rows with 1–5 numeric missing values 91.54%, Commercial
63.07%. Those overlapping partitions motivated the compact screen, but did
not justify an automatic cold model or clipping. Complete counts, MAE, RMSE,
SSE, fit-derived thresholds and segment membership definitions are in the raw
evidence archive.

Raw archive: `round4-regression-evidence.zip`, 15,675,234 bytes, 496 entries.
SHA256: `dd9bff617e9d79d55dc40c0e6408c917c8add9b2512768c18c711dcb42e35adf`.
Its CRCs and every manifest entry were verified before delivery. The archive
is delivered with the task response; source datasets and trained models are
not committed to Git.

## Calibration alignment and adapter behavior

The existing LightGBM diagnostic stopped at 31 trees and refitted at the
existing minimum of 50. The aligned experimental path trained one additional
earlier-only 50-tree model (about 0.282 s) for calibration. Final predictions
were identical because the frozen history schema was unavailable. This proves
the extra fit is counted; it does not establish a benefit to history selection.
No expanded history grid, rolling-fold search, residual correction or target
transformation was introduced.

The new `round4_regression_v1` family fits a separate feature adapter inside
each internal training fold and again on the complete supplied fit. It uses
copied frames and context, declared roles, at most 16 categorical columns and
512 levels per column. This dataset adds three frequency columns. Oversized
maps abstain, unseen levels map to zero. Raw categories remain available. The
old recipe family and all pre-existing runtime files retain their base bytes.
The new family is not registered in the production host.

## Full refit, integrity incidents and resources

The same frozen original source, instrumented source and known-period-5 recipe
were reused for full TRAIN refit. An identical technical replay was accepted:

- Fresh internal fit 35,000 rows; fresh final fit **53,308 rows**; **22,449**
  unlabelled TEST predictions, exact ID order, finite values and valid receipt.
- Two physical fits; host total **11.581724 s**, worker **10.044362 s**.
  Preparation 1.554053 s, internal phase 3.150759 s, full refit phase 1.861374 s.
- Fit-only stopping selected 180 trees. Re-estimation on the full supplied fit
  is part of the unchanged algorithm; it is not a post-Year5 hyperparameter edit.
- Original source SHA256:
  `3cd3690b15edc7f87e5d46909264db7124b5c65af02c26b090b35fdea27ece05`.
- Instrumented source SHA256:
  `2838224b3bcc7c90ef7f69b7047ea3d6ed4e69646cb15c83fbd5bd06fba49042`.
- Recipe SHA256:
  `145900a25d06c218987d781fad81afb8c0bf9cf709a322d9ca24bc7e1a108bcf`.
- Accepted experimental output SHA256:
  `04d723046648abaa51fe32ab7c721206b21524ec2738d1a02603be1a157d115f`.
- Receipt SHA256:
  `608cbedd9463e5e5abdcae974e94714c90bea4e7e55bbb6ba3024862c0060cf1`.

The original B7 run and original full-refit attempt each completed two physical
fits, then failed the host input-hash check. Their worker `fit.csv` files became
exact prefixes of the intended files (9,361 of 22,054 complete rows; 35,496 of
53,308 rows). Predict inputs and frozen recipes remained intact. The fit
ledgers record the original full row counts, so truncation followed model
reads. The cause is unproven. The guard correctly rejected the outputs; it was
not bypassed. Each failure was preserved and repeated **once** with identical
algorithm/budget in a fresh directory. Both replays passed independent input,
output and receipt checks. Failed outputs contribute no quality result.

This is successful contract validation of an experimentally rejected algorithm,
not a recommendation to submit its TEST CSV. TEST targets were neither available
nor sought. The final native 900-second admission was not run because the new
candidate failed quality compatibility.

Accounting: **19 public invocations**, 17 accepted and 2 rejected; **37 started
and completed physical fits**, including 4 fits from rejected invocations.
Their host wall times sum to **158.563014 s**, excluding environment setup,
discovery, tests and analysis. Separately, the real flat-package probe used 3
small invocations / 6 physical fits. No claim is made about total test-suite
fit count. The process monitor reported `RuntimeError: process inspection
unavailable`; memory use, CPU time and cleanup remain unknown despite the
requested 8 GiB setting. The 11.58 s observed full refit is not a universal
scheduler reservation or a prediction for other CPUs/datasets.

## Validation and evidence

31 new round-4 tests and 78 existing rescue regression tests passed. The real
flat-package probe ran outside the source repository, with no `agent/`
directory, and checked imports, fresh full refit, finite predictions, sidecar
hashes, input/context immutability and predict batch/order invariance. Its
report SHA256 is
`3fafbc03ad8b2b63f8904d9d2fd7a70b5d6bb34fb12fe3a0e3f81547b6a5de0e`.

Independent audits recomputed development/known5 metrics from saved CSVs,
checked exact original TRAIN/TEST values and order, same known5/full recipe and
source, receipts, nonces and physical fit ledgers. The final audit SHA256 is
`8630627178a997c9516ea8f21bf71e87767fb7705517a2b96b7da7eb3889dbce`.
The raw archive includes
these audits, per-run summaries/segments/predictions, recipes, source wrappers,
environment inventory and failed-attempt evidence. It excludes source datasets,
worker feature copies and trained models. SHA256 manifests cover its contents.

Environment: Python 3.12.14 and declared core platform package versions, including
CatBoost 1.2.8 and LightGBM 4.6.0. Optional AutoGluon/Torch were not installed for
this isolated candidate experiment. A truncated CatBoost installation initially
failed to import; reinstalling the exact declared version fixed it before any
recorded model fit. Repository dependency files were not changed.

The useful result is a rejected feature hypothesis with a reproducible failure
on new categories, plus evidence that the current history-aware incumbent
remains stronger. Any next experiment needs a newly declared development
hypothesis and must retain this negative result.
