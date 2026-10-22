# External benchmark laboratory

`research_lab` is a development evaluator for public tabular datasets. It stages ordinary agent tasks, runs two frozen CPU checkouts serially, audits saved predictions independently, and reports every planned outcome. It does not change the contest solver or submit anything to a platform.

There are two separate deliverables: a tested tool and actual measurements. Tests, generated manifests, dry runs and synthetic fixture smoke establish the tool's mechanism. Only executed public-data runs produce development measurements. Local wins do not establish a higher contest score.

## Frozen first comparison

| Arm | Exact commit | Native arguments |
| --- | --- | --- |
| Incumbent | `a45861d7e5588288db66f9ae088a9726f9597370` | `--mode offline --research-search adaptive --search-profile legacy` |
| Candidate | `3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee` | Same common arguments, plus `--finalist-selection --stability-splits 2` |

Both checkouts must be clean. `prepare` checks the exact commits, sources, selected Python/dependencies and inputs before sealing a new output directory. It also pins the separate trusted laboratory/auditor root. Commit the laboratory implementation before preparing an executable plan. Changing source, configs, splits, datasets, dependencies or budget requires a new plan; do not edit a frozen plan to make it pass.

The default pilot is **four pairs / eight arms**, one seed (`1729`), 360 seconds per arm, eight computation threads and a configured child-tree RSS budget of 24 GiB. The suite defaults to seeds `1729, 1730`. The requested PC is Windows 10, i9-10900KF, 32 GB RAM, RTX 3080. Those are intended workstation specifications, not an assertion about the host on which a report was generated. The recorded environment and hardware describe each actual plan.

Pilot accepts exactly one seed; use `suite` for a multi-seed plan. The frozen CPU solvers internally use eight threads, so executable CPU plans require `--threads 8`; smaller values are rejected before any work is prepared.

The worker deadline starts before imports and covers startup, prior/fallback, preprocessing, model search, INNER, outer, full refit, publication and evidence. Only the remaining original deadline reaches the native CLI. The adaptive coordinator must retain more than 300 seconds after imports for a pilot arm; a 20-second fixture cannot establish this path. Native search defaults and queues stay unchanged. Cleanup is measured separately with a 20-second planning allowance per arm; the allowance is not extra training time.

## Reuse and ownership

| Existing mechanism | Laboratory reuse |
| --- | --- |
| `benchmark_maximize.py` and its protocol | Resource qualification and explicit provisional/censored outcomes; its synthetic-only runner is not presented as a real-data runner. |
| `qualify_unseen_tasks.py` | Read-only solver identities, sanitized process supervision and independent CSV scoring. Its own duration guard remains unchanged. |
| `benchmark_agent.py` | Exact current `run.json`, frozen publication identity, receipts and checkpoint linkage. |
| `agent/task_spec.py`, `agent/metrics.py` | Task schema, exact class/probability semantics and trusted scoring. |
| `agent/resources.py`, `agent/execution.py` | Existing process/resource behavior is preserved; laboratory adapters retain unavailable observations and measure cleanup separately. |
| Frequency experiment protocol | Development/confirmation separation, frozen methods and retention of losing or incomplete outcomes. |

Only `research_lab/**`, `tests/test_research_lab_*.py` and this document belong to the laboratory. The auditor imports the pinned trusted root, never Python code from a supplied candidate checkout. Candidate code executes in a fresh child with a sanitized import path. This is a data-flow protocol, not a filesystem or network OS sandbox; the frozen offline code and actual API reports are audited. Sampled RSS is not an OS memory sandbox. The runner's environment excludes API credentials, proxies and developer import paths, sets native thread limits and hides CUDA devices for the CPU lane. Actual report API attempts/responses are checked; an empty API environment alone does not prove zero calls.

## Public data and leakage rules

The catalog is [datasets.json](../research_lab/datasets.json). Each entry records official provenance/license URLs, attribution, exclusions, split policy and prior exposure. Ready catalog metadata is not proof that bytes are cached: `fetch` records actual retrieval time, byte count, content SHA-256 and adapter metadata in separate receipts. A locally calculated SHA is not a publisher signature.

| Dataset | Metric | Planned role |
| --- | --- | --- |
| Bank Marketing | LogLoss | Development pilot; pre-call prediction excludes `duration`. |
| WDBC | ROC AUC | Development pilot. |
| Red Wine Quality | RMSE | Development pilot; red variant is fixed. |
| Auto MPG | RMSE | Development pilot; missing-value handling is fold-local. |
| Dry Bean | Macro F1 | Confirmation candidate. |
| Energy Efficiency | RMSE | Confirmation candidate; unused second target excluded. |
| Bike Sharing | RMSE | Confirmation candidate; target components excluded and time ordering retained. |
| Adult | ROC AUC | Confirmation candidate. |
| Iris | Macro F1 | Known classic sanity case; not independent confirmation. |
| Abalone | MAE | Confirmation candidate. |

Confirmation is procedural: publicly available datasets may already be familiar to developers or models. Record prior exposure honestly. Do not tune policy on confirmation labels/scores or send them to the policy developer before exact candidate/config freeze. Repeated seeds or targets from the same source do not become independent datasets. Existing contest public TRAIN and synthetic fixtures remain separate sanity/mechanism evidence.

The current fetcher supports bounded official unarchived UCI CSV downloads only. ZIP/tar and other archive sources are rejected rather than extracted. There is no Kaggle login or user-key requirement. Unknown license, unavailable source, unsupported schema and unverified provenance remain explicit outcomes. `fetch` is the only network command. A corrupt or missing cache blocks preparation/running; the runner and re-auditor never repair it by downloading.

Adapters perform static schema conversion and exclude targets, known target-derived columns and source IDs before writing the solver task. Generated IDs are exact strings and never model features. Grouped duplicate units stay together; time tasks use earlier train and later audit according to the sealed rule. Imputation, scaling, categories, feature selection and target statistics belong inside each solver fit, not the adapter. Audit labels live outside solver task directories and are absent from worker config, argv, environment, task text and sample submission. Literal labels and probability class order remain explicit.

## Windows PowerShell

Run from a clone containing the published laboratory branch. These commands install dependencies once; the runner never invokes `pip`. Keep the two solver worktrees separate from the laboratory checkout and outputs.

```powershell
git fetch origin
git switch work/external-benchmark-lab-v1
git worktree add --detach ..\alfa-incumbent a45861d7e5588288db66f9ae088a9726f9597370
git worktree add --detach ..\alfa-candidate 3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_research_lab_*.py" -v
.\.venv\Scripts\python.exe -m research_lab catalog --manifest research_lab/datasets.json
.\.venv\Scripts\python.exe -m research_lab fetch --manifest research_lab/datasets.json --profile pilot --cache data/lab-cache
.\.venv\Scripts\python.exe -m research_lab estimate --manifest research_lab/datasets.json --profile pilot --cache data/lab-cache --time-budget 360 --threads 8 --memory-gib 24 --dry-run
.\.venv\Scripts\python.exe -m research_lab prepare --manifest research_lab/datasets.json --profile pilot --incumbent-root ..\alfa-incumbent --candidate-root ..\alfa-candidate --candidate-commit 3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee --python .venv\Scripts\python.exe --cache data/lab-cache --output data/lab-runs/pilot-001 --time-budget 360 --threads 8 --memory-gib 24
.\.venv\Scripts\python.exe -m research_lab run --output data/lab-runs/pilot-001
.\.venv\Scripts\python.exe -m research_lab re-audit --output data/lab-runs/pilot-001
.\.venv\Scripts\python.exe -m research_lab report --output data/lab-runs/pilot-001
```

Quote executable/root paths containing spaces or Cyrillic text. PowerShell uses the call operator for a quoted executable, for example `& "C:\ML Проект\.venv\Scripts\python.exe" -m research_lab catalog`. No virtualenv activation is necessary. Unit tests for Unicode paths and Windows branches do not establish that the full native series ran on Windows; actual Windows execution must be recorded separately.

If `py` opens a Windows file-association dialog, inspect `Get-Command py -All`
and invoke the verified launcher by its full executable path. For example,
`& "C:\Windows\py.exe" "-3.12" -m venv .venv` bypasses an extensionless file
named `py` earlier in command lookup.

The worker retains the selected interpreter's site-packages even when `.venv`
is inside the auditor checkout. Earlier laboratory revisions incorrectly removed
all import paths under the checkout, causing an immediate `ModuleNotFoundError`
despite successful package preflight. Preserve affected failed plans and use a
fresh output after the fix. To retain their exact frozen auditor as well, create
a separate detached auditor worktree for the corrected revision and reuse the
already installed virtualenv by its absolute `Scripts\python.exe` path. Package
reinstallation is unnecessary when its recorded versions are unchanged.

## Linux / Bash

```bash
git fetch origin
git switch work/external-benchmark-lab-v1
git worktree add --detach ../alfa-incumbent a45861d7e5588288db66f9ae088a9726f9597370
git worktree add --detach ../alfa-candidate 3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p 'test_research_lab_*.py' -v
.venv/bin/python -m research_lab catalog --manifest research_lab/datasets.json
.venv/bin/python -m research_lab fetch --manifest research_lab/datasets.json --profile pilot --cache data/lab-cache
.venv/bin/python -m research_lab estimate --profile pilot --cache data/lab-cache --time-budget 360 --threads 8 --memory-gib 24 --dry-run
.venv/bin/python -m research_lab prepare --manifest research_lab/datasets.json --profile pilot --incumbent-root ../alfa-incumbent --candidate-root ../alfa-candidate --candidate-commit 3b69d50bd7338b9180aaf7d4cb9a101a2a3435ee --python .venv/bin/python --cache data/lab-cache --output data/lab-runs/pilot-001 --time-budget 360 --threads 8 --memory-gib 24
.venv/bin/python -m research_lab run --output data/lab-runs/pilot-001
.venv/bin/python -m research_lab re-audit --output data/lab-runs/pilot-001
.venv/bin/python -m research_lab report --output data/lab-runs/pilot-001
```

The pinned runtime subset is `requirements.txt`; the supplied full platform list remains `requirements.platform.txt`. AutoGluon/XGBoost presence or absence must be identical for both CPU arms and is recorded in versions. Missing required packages block preflight. Do not silently install another version, change the official dependency files or add packages during a measured run.

## Stage A fixture and safe continuation

Run fixture smoke before public-data measurements. The example below uses one synthetic task, a 60-second test-only cap and two incumbent checkouts. It checks the native short-budget AutoML route; the real pilot checks adaptive v3. The native CLI selects its own route from the remaining original budget. The lab does not force v3 for a short fixture.

```powershell
git worktree add --detach ..\alfa-incumbent-copy a45861d7e5588288db66f9ae088a9726f9597370
.\.venv\Scripts\python.exe -m research_lab prepare --profile fixture-smoke --incumbent-root ..\alfa-incumbent --candidate-root ..\alfa-incumbent-copy --candidate-commit a45861d7e5588288db66f9ae088a9726f9597370 --self-check --python .venv\Scripts\python.exe --cache data/lab-cache --output data/lab-runs/fixture-001 --time-budget 60 --threads 8 --memory-gib 24
.\.venv\Scripts\python.exe -m research_lab run --output data/lab-runs/fixture-001
.\.venv\Scripts\python.exe -m research_lab re-audit --output data/lab-runs/fixture-001
.\.venv\Scripts\python.exe -m research_lab report --output data/lab-runs/fixture-001
```

Bash uses the same arguments with `.venv/bin/python` and `../` worktree paths. No fetch is required for the fixture. Use a new named output directory for every new plan or deliberate repeat.

The short fixture's typed `automl` evidence must bind the exact returned `automl_report.json`, input snapshot, selected model config, outer predictions, full-TRAIN refit phase, checkpoint and published CSV. Native v3 evidence keeps its own receipt checks. AutoML evidence is accepted only for `fixture-smoke`; it cannot satisfy the real adaptive pilot's v3 contract. Missing nested execution observations remain provisional, and failed attempts with unknown cleanup remain censored.

Use `report.json` → `fixture_gate.passed` as the fixture prerequisite for a new pilot. It requires a completed, independently re-audited mechanism for every planned fixture arm, fully observed nested/outer resources and verified actual API 0/0. Inspect `mechanism_complete`, `resources_observed` and `zero_api_verified` when the gate is false. Fixture arms always have `quality_verified=false`, `qualified_cpu=false`, `quality_evidence=false` and `quality_admission=false`; zero `qualified_cpu_arms` is expected. A fixture is neither an adaptive quality test nor evidence of a policy improvement. Keep old plans/results with their original frozen auditor; this audit change requires a fresh plan, not edited saved evidence.

For the native AutoML route, API zero requires the exact pinned native report to declare `api.used=false` and integer `api.calls=0`. Zero calls implies zero attempts/responses; the saved provenance is `native_offline_zero_calls` with its source report hash and native call count. Missing counters are not converted to zero. For v3 the existing native attempts/responses checks remain unchanged. Do not resume an old fixture under this changed method: prepare a fresh named fixture, run/re-audit/report it, inspect `fixture_gate.passed`, then prepare a fresh pilot. `resume` remains only for untouched arms of the same unchanged sealed method.

Each arm has a unique run ID and a state journal: `prepared → started → finished/failed/censored`. Started is written atomically before spawn. Terminal `state.json` has a separate `state.sha256` seal. Run and machine locks exclude competing coordinators. A failed/timeout arm without confirmed descendant cleanup becomes censored and stops the series; unavailable RSS/CPU remains unknown, never zero. A successful audited CSV with incomplete resource observation is provisional.

After a process interruption, inspect states and `run_report.json` first. When cleanup is proven and seals are unchanged, resume can start only untouched arms:

```powershell
.\.venv\Scripts\python.exe -m research_lab resume --output data/lab-runs/pilot-001
```

```bash
.venv/bin/python -m research_lab resume --output data/lab-runs/pilot-001
```

An already-started arm is never automatically retrained. Unknown cleanup requires manual inspection; `resume` does not waive that gate. Retain interrupted/censored evidence. A deliberate retry gets a new output/plan rather than replacing the old outcome.

## Reports, evidence and comparison rule

`report` calls the saved-artifact `re_audit` again. It does not trust a previously printed score or an embedded success flag. Re-audit checks plan/source/config/cache/input/audit seals, the exact current run report, output IDs/schema, native API counts, frozen outer/full-refit receipts, fresh nonces, and byte identity of submission, checkpoint and full-refit output. It performs no training, download, package installation or artifact repair.

Each case reports metric/direction, both arm scores, signed difference, timings, finalization/CSV/cleanup/resource status, and reason. Failed, skipped, untouched, interrupted, provisional and losing cases remain visible. For minimize metrics, signed difference is `incumbent − candidate`; for maximize, `candidate − incumbent`. A win/loss/tie needs both independently verified completed arms with fully observed CPU resources and matching metric semantics. The frozen tie tolerance is recorded in the plan (default `1e-12`). Provisional differences are shown separately and never counted as wins. Fixture/equal-checkout checks never produce policy wins.

The one-seed pilot keeps denominator **four planned pairs**, including failures/censors/missing outcomes. Different raw metrics are never averaged. Different budget tiers have separate summaries; a mixed-tier report has no pooled win rate. Per-class audit details remain in JSON. Unknown API counts remain unknown: zero is reported only when actual pinned reports cover all started arms included in the total.

Outputs are `report.json`, `report.md`, `report.sha256` and `report.md.sha256`. `content_sha256` fingerprints the canonical JSON payload before its own hash field is added; the `.sha256` sidecars hash the actual saved files. Full raw evidence stays under the plan's output directory. Share the compact reports, copied manifest, versions, hashes and exact commands first; large data/model/log bundles are separate. Keep them out of Git. The production archive must be checked through the existing builder to ensure laboratory code/data/models are absent; building an archive is not a platform submission.

The Python API `research_lab.report.report(output, write=False)` returns the same report data without writing files. Normal CLI `report` writes only report outputs, not original receipts or predictions.

## Cost and later profiles

Always run `estimate --dry-run` before an explicitly selected larger profile. It reports arm count, training upper bound, cleanup allowance, known cache bytes, uncached maximum bytes, thread/RAM configuration and additional disk uncertainty. Model size is unknown before fitting. Direct local rental cost is zero; electricity cost is unspecified.

| Example | Training cap alone |
| --- | --- |
| Four pilot datasets × one seed × two arms × 360 s | 48 minutes; plus up to 160 seconds planned cleanup and measured preparation. |
| Ten datasets × three seeds × two arms × 900 s | 15 hours; plus cleanup/preparation. |
| Two final datasets × two arms × 10,500 s | 11 hours 40 minutes; plus cleanup/preparation. |

`suite`, `confirmation` and `final` never run automatically after installation or pilot. The default suite uses the manifest's cases and the explicitly sealed seed list; pass `--seeds` when defining another planned list. Fetch the selected source profile separately before preparation. A changed 360/900/1800/10,500-second tier gets a new equal-budget plan before examining confirmation outcomes. No server rental, paid API or other expense is part of this tool.

## Opt-in GPU development lane

GPU exploration is separate from CPU qualification and gated on a terminal, independently re-audited four-case CPU pilot with proven cleanup. The current lane fits a small fixed CatBoost GPU baseline, not a second AutoML implementation. It records device/driver/library capability, exact inputs/config/seed/budget, predictions and independent audit. Missing driver, CUDA, library support or cleanup produces an explicit skip. It installs nothing and downloads no pretrained models.

```powershell
.\.venv\Scripts\python.exe -m research_lab gpu --output data/lab-runs/gpu-001 --cpu-plan data/lab-runs/pilot-001 --enable --time-budget 120
.\.venv\Scripts\python.exe -m research_lab gpu-re-audit --output data/lab-runs/gpu-001
```

```bash
.venv/bin/python -m research_lab gpu --output data/lab-runs/gpu-001 --cpu-plan data/lab-runs/pilot-001 --enable --time-budget 120
.venv/bin/python -m research_lab gpu-re-audit --output data/lab-runs/gpu-001
```

Use a new GPU output directory. The lane is labelled `exploratory_gpu`; GPU success does not qualify a CPU contest policy. Transferring a hypothesis requires a new frozen native candidate and a new paired CPU comparison. A successful GPU import/capability probe or mocked GPU test is not a model-fit result. GPU reductions may be nondeterministic. No actual RTX 3080 measurement is claimed merely because these commands exist.

## Measured execution ledger

The published measurement checkpoint is `c473aa62ec590709ae49cf32a61a144df0351faa`, with the exact same tree (`802662b157b05c978772b6b34570f7f81fb586dc`) as the locally frozen auditor `356e8c71baf901d5535bd83e97c1d5207898a436`. Later delivery changes canonicalize three Windows test fixtures and add reports/documentation; they do not retrain or replace this measured method. The original auditor checkout is retained at `/workspace/scratch/de9281f42bd1/alfa-external-benchmark` so its sealed plans remain re-auditable. Run these saved plans from that frozen root, or restore its source-equivalent checkpoint at the recorded paths; running a different method correctly triggers source drift.

| Stage | Actual result |
| --- | --- |
| A: laboratory tests | 100 pass normally and under `python -O`; no downloads/LLM/benchmark fits in these tests. Three Windows temporary-path fixtures use canonical paths. |
| A: native fixture | `fixture-002`: two actual native CLI exits 0, two valid 80-row CSVs, read-only re-audit valid with no integrity errors. Candidate full outer/refit/publication chain verified; short-budget adaptive quality is not admitted. The earlier `fixture-001` exposed transient artifact indexing and remains preserved as a failed integrity smoke. |
| B: real pilot | Four pairs/eight arms planned at 360 s, seed 1729. Bank incumbent executed; one internal trial timed out with unknown cleanup. One arm censored, seven never started, **0/4 completed pairs**, zero candidate starts. |
| C: bounded suite / confirmation | Zero model runs. All ten sources downloaded and their contracts/splits preflighted; this is not confirmation measurement. |
| D: 10,500-second CPU qualification | Not run. |
| Native Windows ML series | Not run in this Linux executor. Windows commands and process-path tests are supplied; CI status is stated separately. |
| GPU | Actual capability probe: CatBoost 1.2.8 reports 0 devices; `nvidia-smi` absent. CPU gate also remains censored. Four explicit skips, **0 GPU model fits**. GPU saved-artifact re-audit passed. |

Actual runtime: Python 3.12.14, the 15 pinned `requirements.txt` packages, AutoGluon/XGBoost absent identically for both prepared arms. The container had 8 CPU quota and 8 GiB total cgroup memory, although the child RSS preset remained the workstation's 24 GiB. RSS/CPU inventory and descendant cleanup were unavailable; this is development evidence, not official CPU resource qualification. Hardware/version details are in [summary.json](../research_lab/results/summary.json).

| Dataset / seed / tier | Metric | Incumbent diagnostic CSV score | Candidate | Outcome |
| --- | --- | ---: | --- | --- |
| Bank Marketing / 1729 / CPU 360 s | LogLoss, minimize | 0.7231302207780654 | Never started | Censored; no quality/policy admission. |
| WDBC / 1729 / CPU 360 s | ROC AUC, maximize | Never started | Never started | Not completed. |
| Red Wine / 1729 / CPU 360 s | RMSE, minimize | Never started | Never started | Not completed. |
| Auto MPG / 1729 / CPU 360 s | RMSE, minimize | Never started | Never started | Not completed. |

Bank's saved CSV has 9,043 rows; outer receipts prove 28,936 fit / 7,232 validation rows and full refit proves 36,168 fit / 9,043 test rows. Exact source/recipe/context/nonces and byte-identical full-refit/checkpoint/published CSV passed independent checks. The native CLI returned 0 with adaptive coordinator active and actual API attempts/responses **0/0**. Worker wall time was 269.1723 s; measured cleanup was 0.00105 s but its completeness is unknown. Internal `baseline_search` candidate 14 (`refine_6`, CatBoost depth 10 / 400 iterations / learning rate 0.03) timed out after 10.02 s, exit −9, with unknown cleanup. The censor guard therefore stopped the series despite the valid final CSV. No retry, new budget or selective continuation was used.

Wins/ties/losses: 0/0/0; denominator four planned pairs, one censored and three incomplete pairs. A candidate improvement estimate is unavailable. Do not read the diagnostic Bank metric as a qualified baseline or a candidate win.

Artifacts:

- [Paired report](../research_lab/results/pilot-001.md), [JSON](../research_lab/results/pilot-001.json), [GPU report](../research_lab/results/gpu-001.json).
- [Source/adapter/split fingerprints](../research_lab/results/source-split-fingerprints.json): all ten official sources, actual bytes/SHA and prepared split hashes; no targets or confirmation scores. This enriched snapshot is evidence-only; use `research_lab/datasets.json` as the input to `fetch`/`prepare`.
- [Summary and versions](../research_lab/results/summary.json), [file hashes](../research_lab/results/SHA256SUMS).
- Report SHA-256: `655139a5701fce94501d9e92686807b554cfa471dc07e65802188d6e00d07d00`.
- Source/split manifest SHA-256: `96350492c3c0f4fd0606e55b3b39b549a38b48ba98123fc9898710655dcab75d`.
- Full local evidence bundle: `/workspace/scratch/de9281f42bd1/alfa-external-benchmark/data/lab-evidence-bundles/external-benchmark-evidence-v1.tar.gz`. Code/history bundle is stored alongside it. Raw data/labels/models/logs are excluded from Git and the contest archive.

The saved pilot is censored: do not mechanically resume it or erase its outcome. On the Windows PC, run the commands above with a fresh named plan; only after verified CPU cleanup does the opt-in GPU fit gate open. No paid API, rental or platform submission was performed.
