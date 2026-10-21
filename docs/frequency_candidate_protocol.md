# Frequency candidate v1: frozen protocol

This experiment is isolated from runtime defaults. Base: `143708b91f50959052dceb15ac968ef279071f02`; assignment: `work/round4-frequency-task` at `c44fe22455e90a912f7da9bfbf06afc2f0ab2017`. Branch: `work/frequency-candidate-v1`. Exactly the seven files named in the assignment are added. No external model APIs, contest packages or platform submissions are used.

## Candidate fixed before quality measurements

For each declared categorical column, append `log1p(count_fit(value))`; retain original features. Unknown values produce zero. Literal strings, including `NA`, `0007`, `7`, whitespace and the empty string, are exact. Actual Python missing scalars share a separate missing category. Native CSV transport reads every cell as a literal string with NA inference disabled, so an empty CSV field is the empty string at that boundary; the experiment cannot establish distinctions erased by the source CSV format.

Counts use only the current invocation's fitting rows. Every INNER, outer and full refit constructs a fresh dictionary. No target input is accepted by the transform. More than 16 declared categorical columns or 4096 observed categories in any column rejects the entire candidate before its model fit. Missing counts toward that limit. There is no selection or truncation. Names are `__frequency_log1p_` plus the first 16 SHA-256 hex digits of the exact column name. Any schema collision rejects the candidate. Inputs, context and the complete base recipe are copied; class order and output protocol remain the baseline's.

The wrapper embeds the actual frozen baseline source and preserves its complete recipe, including components. It extends the baseline's numeric/features metadata and retains categorical metadata. A native invocation writes a diagnostic `frequency_evidence.json` only after the baseline returns. Its dictionary, typed augmented-frame hashes, phase, nonce, code/recipe hashes and embedded baseline hashes are independently reconstructed during audit. This trace is not a replacement for the host execution receipt.

## Current v3, equal inclusive budgets

Both arms call the current `run_agent_v3(task, client=None, local_only=True)` with the ordinary default search settings. The experimental driver temporarily wraps `continue_locally`: after native startup, baseline preparation and its INNER measurement, the candidate arm proposes exactly one eligible frequency wrapper through the real `_assess_candidate`. The original continuation then runs unchanged in both arms. Typed admission and the host's measured cost/refit feasibility remain authoritative. The wrapper does not set accepted/best, change queues, guards, finalization or defaults. No categories means skip without another fit.

Each synthetic arm has 90 seconds from native child start, including imports, startup, preprocessing, fits, continuation, outer, fallback, full refit and publication. The parent enforces the same hard cap and passes the remaining budget to the host, retaining its final reserve. The synthetic coordinator has 1800 active seconds; a pair starts only with at least 180 seconds remaining. Incomplete/censored pairs remain failures, never shortened comparable arms. A timeout with unverified cleanup stops the series. Python process-tree RSS and a 32 GiB OS limit are not promised where this executor cannot prove them.

Only one training arm runs at a time. All agents in this session pause other model fits for the quality window. The executor is checked for load before that window. No agreement with absent collaborators is inferred; evidence of competing training would invalidate timing conclusions and stop the series.

## Frozen cases, order and seeds

`frequency_candidate_cases.py` is the executable definition. Every case has 1200 training and 300 independent audit rows. Four numeric features are `measurement`, `auxiliary`, `noise`, `scale`; the categorical feature, where present, is `habitat kind`. Literal class labels are `01`, `NA`, `class c`; IDs and targets never become features. Audit truth stays outside solver task directories and child configuration.

The category generator has 192 known and 16 unknown tokens. Its latent prevalence is proportional to `1/(rank+1)` over 208 ranks, independently sampled for train and audit; audit is a 90/10 known/unknown mixture. Ordinary known token spellings at ranks 6–191 are shuffled independently of rank. Six literal edge tokens occupy fixed ranks 0–5; unknown token spellings are fixed. Classification noise replaces 6% of labels with probabilities `[.70,.25,.05]`. Targets never use empirical audit/test counts.

| Stage | Case | Seeds | Target and validation |
| --- | --- | --- | --- |
| Development | category_prevalence_classification | 701 | Rank bands `<12`, `<128`, otherwise; 6% label noise; rare class; duplicated raw training units kept in grouped validation |
| Development | category_no_frequency_signal | 701 | Numeric `digitize(x0 + .65*x1 - .25*x2 + N(0,.35), [-.5,1.1])`; categories independent of target; grouped duplicates |
| Development | category_prevalence_regression | 701 | `3*log(latent_prevalence) + 2*x0 + .8*sin(x1) + N(0,.45)`; random validation |
| Development | numeric_only | 701 | Same numeric classification mechanism, no categories; grouped duplicates; explicit candidate skip |
| Confirmation | category_prevalence_classification | 1907, 7919 | Same frozen mechanism and grouped policy |
| Confirmation | category_no_frequency_signal | 1907, 7919 | Same frozen mechanism and grouped policy |

Case order is development rows above, then confirmation case/seed order in the prepared plan. Pair order alternates: control then frequency for even zero-based pair indices, frequency then control for odd indices. Plan preparation freezes exact input files, audit truth, source/config hashes, environment, outer row IDs and native INNER split IDs before results. A separate confirmation seal is written after all development pairs and before any confirmation arm. No code, cases, policies, model options, seeds or budgets may change between these stages. Protocol smoke and fake-worker tests provide no quality evidence.

## Public TRAIN supplement

If real public training data exists, use the existing grouped duplicate folds 0 and 1, with 300 seconds per arm and a separate 1400-second coordinator cap. This known dataset is not confirmation. Do not download data, infer hidden answers or substitute synthetic data. Missing data is recorded as `skipped_missing_data`, with checked absolute paths. Initially checked task paths:

- `/workspace/scratch/de9281f42bd1/alfa-frequency-candidate/data/classification/task.md`
- `/workspace/scratch/de9281f42bd1/alfa-oof-calibration/data/classification/task.md`
- `/workspace/scratch/de9281f42bd1/alfa-night-unseen/data/classification/task.md`

## Evidence and decision

Audit verifies native startup, exact INNER fit/predict rows, fit-only counts and immutable inputs, typed host feedback, source/recipe/nonce/receipt/output hashes, actual outer validation and full-training refit, and byte identity between published CSV, last-valid checkpoint and its matching refit result. External metrics and per-class Macro F1 harm are recomputed from the saved CSV. Baseline code/recipe equality is reported per pair: mismatch permits only a full-agent comparison, not a pure frequency attribution. Equal baselines still permit time-allocation effects.

Read-only re-audit must not train or repair artifacts, including under `python -O`. Partial pairs, missing refits, tampering and source/environment drift cannot become completed. All losses, skips, failures, censored runs and measured costs are retained. Frozen executable/method hashes and reproduction commands are published in the report. Integration is a separate decision by the main owner; an unsupported or costly candidate is explicitly not recommended.
