# Round6 regression calibration — retain incumbent

Recorded 2026-10-23 MSK. This is an inactive negative development study.
**Decision: retain `rescue_regression_v1` (I).** A20 improved period4 RMSE by
0.1047408%, below the preregistered 0.2% screen threshold. S20 and S50 were
rejected by their internal B SSE gate and returned the exact I predictions.
No finalist was promoted, so confirmation was not run. No candidate passed
all gates; no integration request is made. No new family is activated.

## Immutable identities

| Item | SHA |
| --- | --- |
| Fresh origin/main base | `114fc2c18595898e0f6ec099e42795eec4af0be9` |
| Base tree | `0ee506c852646704037620759dc35ac0fa688072` |
| Clean local measured plan + implementation commit | `d6354a4d6cf446ac895b1ca298527ff8fea7666b` |
| Published plan + implementation commit | `0b721203077c12777e72d4e294b47f7fa9f3dbcf` |
| Equal measured/published tree | `ce3ed45f7ecc85b3017068e7780e849612ed49cd` |
| Measured aggregate dependency SHA256 | `d599fa2cbaa324d1335b3d9e1fdb6d303e7925a69c19defef5ad276ca35491a6` |

The remote branch was read back and its commit fetched locally before any
model fit, including native smoke. `publication_identity()` verified commit
object type, a clean local tree and exact Git tree equality. The 17 original
Round5 rescue/dependency hashes still match; the plan additionally freezes
six existing diagnostic dependencies. All changes are new allowed Round6
files. Measured source/evidence remains immutable; later commits add reports.
The PR description records exact final local/published result HEADs, avoiding
a self-referential commit SHA in this document.

## Fixed screen, period4, cap180

Supplied fit: 22,054 rows. External held: 12,946 rows. CPU8, original-scale
CatBoost RMSE, seed42. All four complete invocations finished below 180s.
Positive gain means lower RMSE than the fresh control.

| Arm | RMSE | Gain vs I | MAE | SSE | Host seconds | CatBoost fits | Analytic fits | Actual correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| I | 46.824893087859 | 0.000000% | 24.866786109882 | 28385019.151878 | 9.125818 | 2 | 0 | incumbent_control |
| S20 | 46.824893087859 | 0.000000% | 24.866786109882 | 28385019.151878 | 11.135661 | 4 | 1 | B_rejected |
| S50 | 46.824893087859 | 0.000000% | 24.866786109882 | 28385019.151878 | 10.941561 | 4 | 1 | B_rejected |
| A20 | 46.775848304478 | 0.104741% | 24.799299187351 | 28325588.880678 | 8.980658 | 3 | 1 | applied |

A20's aligned uncorrected identity ablation equals I byte-for-byte in this
screen (`8b25a82b16a485722d7ccb491105f4fb8d0edaeb79c195261a1859c985d3ee52`).
Both selected 66 trees. Thus the small observed improvement here comes from
the frozen correction; the general A-only stopping/alignment confound remains
explicit in the preregistration. `base`, `mean` and `calibrated` coincide with
schema=None. The original submission CSV and all four ablation CSVs are kept.

## Correction disposition and B evidence

The common internal calibration contains 10,879 rows, assigned before fits
to A=7,553 / B=3,326 (7,464 / 3,277 raw-feature groups). Prefix fit has 11,175
rows. Split source includes no target, ID or row-index value. No raw group is
exported. B is physically absent from correction-tree selection arguments.
Each correction arm paid its A selection, aligned prefix recheck and one
analytic parameter fit. Accepted A parameters were never refitted on A+B.

| Arm | Offset | Effective slope | Identity B SSE | Corrected B SSE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| S20 | -0.175880575385352 | 1 | 11610660.694607794 | 11610671.382987438 | B rejected: SSE worsened |
| S50 | -0.439701438463380 | 1 | 11610660.694607794 | 11611073.239582514 | B rejected: SSE worsened |
| A20 | -0.175880575385352 | 1.016714447998151 | 11610660.694607794 | 11604847.023356415 | Applied once; external gain below promotion threshold |

Affine center: 86.01808333064568; A prediction std: 21.658216616714963;
A target std: 59.11633792383838. No clipping bound was reached. These are
one global correction per arm; no segment-specific correction was fitted.
The common assignment hash is
`fdb7537bcd4086b851b943c014263b6c74088edfe1e47169de5c4abf6dd1a296`.

Completed screen disposition counts: applied=1, B_rejected=2,
insufficient_support=0, no_calibration=0, incumbent_control=1.
`round6_regression_correction_decisions.csv` preserves every arm/fold reason.
Single-period/support fallback was verified model-free; no extra development
models were fitted to exercise it.

The one permitted host INNER-fit discovery used 11,175 rows, 8M/48,
extended=False and returned limited/None (`proposer_limited`). Schema remained
None for all four arms. Internal B found no known significant category/novelty
veto; history and unsupported segments remain **unknown**. This is not history
or cold admission. No confirmation fold was run or treated as passed.

On B rejection, the separately paid unchanged incumbent retains its legacy
full-calibration stopping contract under the original remaining deadline.
This B-label exception applies only after correction rejection and was frozen
before fits. S20/S50 consumed four CatBoost fits each, including fallback.
No deadline reset, partial result or alternate base was called incumbent.

## Verification and transparent audit supplements

- Final frozen-source batch: **20 model-free tests passed**, native test opted
  out of ordinary discovery. Tests cover fit-only preprocessing, A/B isolation,
  labels/IDs, frozen roles/recipes, no mutation, alignment, exact single-period
  I delegation, constant/nonfinite/support fallbacks, clipping/ties/veto,
  deadlines and observed physical/analytic fit ledgers.
- Native flat ZIP: exactly **two S20 invocations**, cap90 each, executed outside
  checkout with no `agent/` directory. Query-row addition, query order and
  training-row order invariance passed. Six actual CatBoost fits and two
  analytic fits completed; full reservation-to-registration wall=4.305932s.
- Required P2 fix: the new writer checks its original deadline after diagnostics,
  source/receipt/output/ledger validation, immediately before success. The
  independent reader rejects over-cap, negative or nonfinite elapsed time.
  Fake-clock cap180/elapsed201 rejects in both paths; elapsed179 admits.
- Built-in audit: all **16 metric tables and 400 segment rows** recomputed.
- Separately authored auditor imports no project modules. It independently
  recomputed all metrics/segment masks from preserved predictions and host
  answers, checked exact IDs, hashes, recipes, caps, actual engine/analytic
  traces and unchanged source/output files. Final status: **completed**;
  completed runs=4, metric tables=16,
  segment rows=400. Submission CSVs are also checked.

Two audit/packaging details are preserved transparently:

1. Independent audit v1 demanded bitwise equality between raw ablation CSVs
   and a second CSV serialization. It failed before declaring any run verified.
   Read-only quantification found max absolute delta 5.684341886080802e-14
   (max relative 2.5175325813520915e-16), 13–18 of 12,946 rows. Auditor v2 only
   changes these cross-serialization comparisons to rtol=atol=1e-12 and reports
   actual deltas. Metric/segment checks remain rtol1e-11, atol1e-9 (SSE atol1e-6);
   IDs and hashes remain exact and deadlines have no tolerance. V1 script,
   failed report, v2 script/report and both charged costs are retained.
2. The measured export helper looks for `frozen/candidate.py` and `recipe.json`,
   while the actual freeze emits `research_candidate.py`, `research_recipe.json`
   and `frozen_candidate.json`. Complete recipes were already in summaries and
   source snapshots in the original export. The final bundle adds these actual
   three immutable freeze files per arm, with hashes, in a packaging supplement.
   It preserves the original export manifest/receipt and does not rewrite the
   measured helper, predictions, metrics or original experiment reports.

No model was rerun for either audit-only correction. Internal A/B prediction
vectors were not retained, so the independent auditor does not claim to
re-score their SSE from raw vectors; frozen parameters, aggregate B evidence,
source boundaries and actual fit traces are preserved. The public held outputs
are fully and independently re-scored.

## Actual costs and retained failures

Actual CatBoost fits: **19 started / 19 completed** (13 screen + 6 smoke).
Analytic correction fits: **5 started / 5 completed** (3 screen + 2 smoke).
Combined fitting operations: **24**. Discovery, validation and audits add no
model fits. Technical model retries=0; failed model invocations=0;
preflight failures=0. API calls=0; GPU=0; CPU thread configuration=8.
Process inventory was unavailable: measured CPU seconds, peak RSS and cleanup
qualification are unknown. Configured 8GiB is not measured memory admission.

The complete serial ledger below includes failed preliminary fixture checks,
the failed independent audit, the successful revised audit, all model work
and evidence export checks. Timers are host wall measurements; nested fit,
preprocessing and observer timers are not added again.

| # | Work | Status | Charged seconds |
| ---: | --- | --- | ---: |
| 1 | prepare: development_inputs | completed | 3.393912 |
| 2 | model_free_checks: preregistered_contract_tests | completed | 2.219277 |
| 3 | audit_only_checks: predecessor_test_accounting_only | failed | 0.287955 |
| 4 | validation: flat_smoke | completed | 4.305932 |
| 5 | discovery: p4 | completed | 2.234713 |
| 6 | screen: p4/I | completed | 9.125818 |
| 7 | screen: p4/S20 | completed | 11.135661 |
| 8 | screen: p4/S50 | completed | 10.941561 |
| 9 | screen: p4/A20 | completed | 8.980658 |
| 10 | audit: all_completed_predictions | completed | 6.525449 |
| 11 | audit_only_checks: independent_metrics_segments_audit | failed | 3.692488 |
| 12 | audit_only_checks: independent_metrics_segments_audit_v2 | completed | 4.685584 |
| 13 | audit_only_checks: data_free_export_manifest_and_crc | completed | 2.584115 |

**Total charged wall: 70.113123s / 7200s ceiling.** The preliminary 0.028s
passing helper tests and 0.177s candidate fixture failure were reported before
source freeze and charged as an accounting-only record; they do not validate
the final source. The corrected final-source test batch is a separate receipt.
The extra CSV quantification cost (0.561158846s) is included in audit-v2 charge.
No screen finalist means periods2/3/4 confirmation was not started. Year5,
new full TRAIN and original TEST prediction were not run. Runtime, builder,
legacy helpers, shared host/CLI, Windows night queue, PR51, app and Murad code
are unchanged. No platform submission or native admission is claimed.

## Evidence package

`round6-regression-evidence.zip` — 7151770 bytes,
SHA256 `4c4baeefa7a1be65b2d14dcdd05ad73e8965184d8acfefbc3748012e61ffe39c`.

The ZIP contains explicit data-free source snapshots, full recipes, receipts,
engine/analytic ledgers, environment/pins, all original output CSVs, segment
reports, original and revised independent audits, publication identities and
SHA256/size manifests. CRC and every manifested member hash were verified.
It excludes raw TRAIN/TEST/answers, fitted models, context/fit/predict input
files and raw A/B group strings. The final compact JSON/CSV supplies the
complete post-packaging cost ledger; the archive cannot contain its own final
hash/cost receipt without circular self-reference.

See `round6_regression_commands.md` for Linux/PowerShell reproduction and the
PR for final publication HEAD mapping. This negative development result does
not establish leaderboard performance or compatibility/native qualification.
